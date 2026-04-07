import copy
import math
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as transforms
import torch
import random
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from scipy.stats import spearmanr
import csv
import os
import matplotlib.pyplot as plt


from modules.base import ModuleType
from modules.operations import *
from modules.learnable import *
from modules.input import *
from graph.architecture import *
from graph.executor import *
from graph.generator import *

class Arena:
    def __init__(self, n_fights=1, architecture_size=16, arena_contestants=3, dataset_size = 256+64,
                  train_test_split= 0.7, generation_type="agnostic", verbose=True, report=False, pcp=0.38,
                    cpu=False, simp_bal=0.36, speed_bal=0.3):
        self.arena_contestants = arena_contestants
        self.tournament = []
        self.n_fights = n_fights
        self.architecture_size = architecture_size
        self.arena_contestants = arena_contestants
        self.dataset_size = dataset_size
        self.train_test_split = train_test_split
        self.generation_type = generation_type
        self.verbose = verbose
        self.report = report
        self.pcp = pcp # parameter complexity penalty exponent
        self.cpu = cpu
        self.simp_bal = 0 # simplicity didn't show any advantage
        self.avg_learn = 0.2197 # estimated over 340 runs, error +- 0.0043
        self.std_learn = 0.7371 # estimated over 340 runs, error +- 0.0067
        self.avg_speed = -1.8779 # estimated over 340 runs, error +- 0.0093
        self.std_speed = 1.3691 # estimated over 340 runs, error +- 0.0124
        self.speed_bal = speed_bal


    def _valid(self, *values, min_val=1e-10):
        return all(math.isfinite(v) and v > min_val for v in values)


    def get_scores(self, arch_1, arch_2, input = None, get_penalties=False, outerfunction="sqrt", randomizeHP=False,
                    pcp=None, uniform=False, input_noise = 0.05, output_noise = 0.01, get_delays=False):
        device = torch.device('cuda' if torch.cuda.is_available() and not self.cpu else 'cpu')

        if device.type=='cuda':
            max_batch_size = 2048
        else:
            max_batch_size = 32
        if pcp is None:
            pcp = self.pcp
        # Generate random values for the hyperparameters
        if randomizeHP:
            random_batch_size = int(max_batch_size + (random.random()-0.5) * max_batch_size * 0.5) # more or less 25% of the batch size
            random_lr = random.choice([0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001,0.00005, 0.00001])
            random_patience = random.choice([6, 8, 10, 12, 15, 20, 25, 30])
            random_min_delta = random.choice([1e-5, 1e-6, 1e-7, 1e-8])
            random_max_iter = random.choice([100, 200, 300, 400, 500, 600, 700])
            train_test_split = random.choice([0.6, 0.7, 0.8])

        else :
            train_test_split = self.train_test_split
        arch_1.reset_state()
        arch_2.reset_state()

        executors = []
        # create the executors  
        executors.append(Executor(copy.deepcopy(arch_1)).to(device))
        executors.append(Executor(copy.deepcopy(arch_2)).to(device))
        # Randomize the weights of the architectures
        executors[0].randomize_weights()
        executors[1].randomize_weights()

        if input is None:
            # create the input
            train_size = int(self.dataset_size*train_test_split)

            input_p = random.randint(1,16)
            input_f = random.randint(1,16)
            if uniform==True:
                base = torch.linspace(0, 1, self.dataset_size).unsqueeze(1).unsqueeze(2)
                base = base.expand(self.dataset_size, input_p, input_f)

                mean = torch.randn(1, 1, input_f) * 2
                std = torch.rand(1, 1, input_f) * 3 + 0.5

                input = (base * std + mean).to(device)
                input = input + torch.randn_like(input)*input_noise
            else:
                input = torch.randn(self.dataset_size, input_p, input_f).to(device)

        train_size = int(self.dataset_size*train_test_split)
        train_input = input[:train_size]
        test_input = input[train_size:]

        # generate the outputs for each executor
        output_1 = (executors[0].forward(input))[0]
        output_2 = (executors[1].forward(input))[0]

        output_1 = output_1 + torch.randn_like(output_1) * output_noise
        output_2 = output_2 + torch.randn_like(output_2) * output_noise

        # Standardize the targets so amplitude/flatness gives no advantage
        std_1, mean_1 = torch.std_mean(output_1, dim=0, keepdim=True)
        std_2, mean_2 = torch.std_mean(output_2, dim=0, keepdim=True)

        # Add epsilon to prevent division by zero for totally dead networks
        output_1 = (output_1 - mean_1) / (std_1 + 1e-5)
        output_2 = (output_2 - mean_2) / (std_2 + 1e-5)

        train_target_1 = output_2[:train_size].to(device)
        train_target_2 = output_1[:train_size].to(device)

        test_target_1 = output_2[train_size:].to(device)
        test_target_2 = output_1[train_size:].to(device)

        # make new executors and fit them
        learner_1 = Executor(copy.deepcopy(arch_1)).to(device)
        learner_2 = Executor(copy.deepcopy(arch_2)).to(device)
        if randomizeHP:
            l1_start_time = time.time()
            learner_1.fit(train_input, train_target_1.detach(), verbose=self.verbose, lr=random_lr, max_iter=random_max_iter, batch_size=min(train_size,random_batch_size), patience = random_patience, min_delta = random_min_delta, cpu = self.cpu)
            l1_delay = time.time() - l1_start_time
            l2_start_time = time.time()
            learner_2.fit(train_input, train_target_2.detach(), verbose=self.verbose, lr=random_lr, max_iter=random_max_iter, batch_size=min(train_size,random_batch_size), patience = random_patience, min_delta = random_min_delta, cpu = self.cpu)
            l2_delay = time.time()-l2_start_time
        else:
            l1_start_time = time.time()
            learner_1.fit(train_input, train_target_1.detach(), verbose=self.verbose, lr=0.01, max_iter=200, batch_size=min(train_size,max_batch_size), patience = 10, min_delta = 1e-7, cpu = self.cpu)
            l1_delay = time.time() - l1_start_time
            l2_start_time = time.time()
            learner_2.fit(train_input, train_target_2.detach(), verbose=self.verbose, lr=0.01, max_iter=200, batch_size=min(train_size,max_batch_size), patience = 10, min_delta = 1e-7, cpu = self.cpu)
            l2_delay = time.time()-l2_start_time

        with torch.no_grad():
            pred_1 = learner_1.forward(test_input)[0]
            pred_2 = learner_2.forward(test_input)[0]
            
            test_loss_1 = torch.nn.functional.mse_loss(pred_1, test_target_1).item()
            test_loss_2 = torch.nn.functional.mse_loss(pred_2, test_target_2).item()

            # Guard against NaN / Inf / zero losses
            if not math.isfinite(test_loss_1) or test_loss_1 <= 0:
                test_loss_1 = 1e10
            if not math.isfinite(test_loss_2) or test_loss_2 <= 0:
                test_loss_2 = 1e10

            # compute the scores
            if outerfunction == "log2":
                K_1 = math.log2(max(2,arch_1.parameter_count()))
                K_2 = math.log2(max(2,arch_2.parameter_count()))
            if outerfunction == "sqrt":
                K_1 = math.sqrt(max(2,arch_1.parameter_count()))
                K_2 = math.sqrt(max(2,arch_2.parameter_count()))
            if outerfunction == "identity":
                K_1 = max(2,arch_1.parameter_count())
                K_2 = max(2,arch_2.parameter_count())
            if outerfunction == "pow1/pi":
                K_1 = max(2,arch_1.parameter_count())**(1/math.pi)
                K_2 = max(2,arch_2.parameter_count())**(1/math.pi)

            deltaK_1 = (K_1/K_2) + math.exp(-K_1*K_2)
            deltaK_2 = (K_2/K_1) + math.exp(-K_2*K_1)
            score_1 = (1/(test_loss_1*( deltaK_1 ** pcp) ))**0.5
            score_2 = (1/(test_loss_2*( deltaK_2 ** pcp) ))**0.5

            del executors[0]
            del executors[0], learner_1, learner_2 #because executors[0] is our previous executors[1] (executors[0] was deleted, which made executors[1] become executors[0]))
            torch.cuda.empty_cache()
            if get_penalties:
                return score_1, score_2, deltaK_1, deltaK_2
            if get_delays:
                return score_1,score_2,l1_delay,l2_delay
            return score_1, score_2

    def occam_selection(self, n_archs=16, verbose=False, randomizeHP=True, use_delays=True, use_cached_norm=True,
                        simp_bal=None, speed_bal=None):
        if simp_bal is None:
            simp_bal = self.simp_bal
        if speed_bal is None:
            speed_bal = self.speed_bal
        generator = Generator(generation_type=self.generation_type)
        architectures = [generator.generate(self.architecture_size) for _ in range(n_archs)]
        simplicity_scores = [0 for _ in range(n_archs)]
        learnability_scores = [0 for _ in range(n_archs)]
        speed_scores = [0 for _ in range(n_archs)]
        
        
        n_fights = 0
        total_pairs =  n_archs * (n_archs - 1) // 2
        for i in range(n_archs):
            for j in range(i + 1, n_archs):
                if verbose:
                    print(f"Fight {n_fights + 1}/{total_pairs}: arch {i} vs {j}")
                
                if use_delays:
                    score_i, score_j, delay_i, delay_j = self.get_scores(
                        architectures[i], architectures[j], randomizeHP=randomizeHP, pcp=0, get_delays=True
                    )
                
                else: 
                    score_i, score_j = self.get_scores(
                        architectures[i], architectures[j], randomizeHP=randomizeHP, pcp=0
                    )
                learnability_scores[i] += math.log((max(score_i,1e-10)))
                learnability_scores[j] += math.log((max(score_j,1e-10)))
                simplicity_scores[i] += math.log((max(score_j,1e-10)))
                simplicity_scores[j] += math.log((max(score_i,1e-10)))
                if use_delays:
                    speed_scores[i]+= math.log(max(1/delay_i,1e-10))
                    speed_scores[j]+= math.log(max(1/delay_j,1e-10))
                if verbose:
                    print(f"score_{i} : {score_i}, score_{j} : {score_j}")
                
                n_fights += 1


        def z_normalize(values):
            mu = sum(values) / len(values)
            var = sum((v - mu) ** 2 for v in values) / len(values)
            sigma = math.sqrt(var) if var > 0 else 1.0
            return [(v - mu) / sigma for v in values]
        
        learnability_scores = [score/(n_archs-1) for score in learnability_scores]
        simplicity_scores = [score/(n_archs-1) for score in simplicity_scores]
        speed_scores = [score/(n_archs-1) for score in speed_scores]
        if use_cached_norm : 
            norm_learn = [(l - self.avg_learn) / self.std_learn for l in learnability_scores]
            norm_speed = [(s - self.avg_speed) / self.std_speed for s in speed_scores]
        else : 
            norm_learn = z_normalize(learnability_scores)
        norm_simp = z_normalize(simplicity_scores)

        if use_delays: 
            occam_scores = [((norm_learn[i]*(1-speed_bal)) + (norm_speed[i]*(speed_bal)))
                         for i in range(n_archs)]
        else : 
            occam_scores = [((norm_learn[i]*(1-simp_bal)) + (norm_simp[i]*(simp_bal)))
                         for i in range(n_archs)]
        max_score_idx = occam_scores.index(max(occam_scores))

        if verbose:
            print(f"average learnability score: {sum(norm_learn)/len(norm_learn)}")
            print(f"average simplicity score: {sum(norm_simp)/len(norm_simp)}")


        return architectures[max_score_idx], occam_scores, max_score_idx, learnability_scores, simplicity_scores
        
    def occam_test(self, ori_architectures, n_archs=8, verbose=False, randomizeHP=True,
                    simp_bal=None, use_cached_norm=True, use_delays=True, speed_bal = None):
        if simp_bal is None:
            simp_bal = self.simp_bal
        if speed_bal is None:
            speed_bal = self.speed_bal
        if not isinstance(ori_architectures, list):
            ori_architectures = [ori_architectures]
        
        generator = Generator(generation_type=self.generation_type)
        architectures = copy.deepcopy(ori_architectures)

        for i in range(n_archs-len(architectures)):
            architectures.append(generator.generate(self.architecture_size))

        learnabilities = [0 for _ in range(n_archs)]
        simplicities = [0 for _ in range(n_archs)]
        speeds = [0 for _ in range(n_archs)]
        
        n_fight = 0
        total_pairs =  n_archs * (n_archs - 1) // 2
        for i in range(n_archs):
            for j in range(i + 1, n_archs):
                if verbose:
                    print(f"Fight {n_fight + 1}/{total_pairs}: arch {i} vs {j}")
                if use_delays:
                    score_i, score_j, delay_i, delay_j = self.get_scores(
                        architectures[i], architectures[j], randomizeHP=randomizeHP, pcp=0, get_delays=True
                    )
                
                else: 
                    score_i, score_j = self.get_scores(
                        architectures[i], architectures[j], randomizeHP=randomizeHP, pcp=0
                    )
                learnabilities[i] += math.log((max(score_i,1e-10)))
                learnabilities[j] += math.log((max(score_j,1e-10)))
                simplicities[i] += math.log((max(score_j,1e-10)))
                simplicities[j] += math.log((max(score_i,1e-10)))
                if use_delays:
                    speeds[i]+= math.log(max(1/delay_i,1e-10))
                    speeds[j]+= math.log(max(1/delay_j,1e-10))
                if verbose:
                    print(f"score_{i} : {math.log((max(score_i,1e-10)))}, score_{j} : {math.log((max(score_j,1e-10)))}")

                n_fight += 1
        learnabilities = [score/(n_archs-1) for score in learnabilities]
        simplicities = [score/(n_archs-1) for score in simplicities]
        speeds = [score/(n_archs-1) for score in speeds]

        def z_normalize(values):
            mu = sum(values) / len(values)
            var = sum((v - mu) ** 2 for v in values) / len(values)
            sigma = math.sqrt(var) if var > 0 else 1.0
            return [(v - mu) / sigma for v in values]


        if use_cached_norm:
            norm_learn = [(l - self.avg_learn) / self.std_learn for l in learnabilities]
            norm_speed = [(s - self.avg_speed) / self.std_speed for s in speeds]
        else : 
            norm_learn = z_normalize(learnabilities)
        norm_simp = z_normalize(simplicities)


        if use_delays: 
            occam_scores = [((norm_learn[i]*(1-speed_bal)) + (norm_speed[i]*(speed_bal)))
                         for i in range(n_archs)]
        else : 
            occam_scores = [((norm_learn[i]*(1-simp_bal)) + (norm_simp[i]*(simp_bal)))
                         for i in range(n_archs)]
        
        occam_scores_sorted = sorted(occam_scores)
        wrs = [0 for _ in range(len(ori_architectures))]
        for tested_arch in range(len(ori_architectures)):
            occam_score = occam_scores[tested_arch]
            rank = occam_scores_sorted.index(occam_score)
            wrs[tested_arch] = rank/(len(occam_scores)-1)

        return wrs, occam_scores, norm_learn, norm_speed

    
    def pareto_selection(self, n_rounds=8, n_archs=10, verbose=False, randomizeHP=True, simp_bal=None):
        if simp_bal is None:
            simp_bal = self.simp_bal
        generator = Generator(generation_type=self.generation_type)
        def z_normalize(values):
            mu = sum(values) / len(values)
            var = sum((v - mu) ** 2 for v in values) / len(values)
            sigma = math.sqrt(var) if var > 0 else 1.0
            return [(v - mu) / sigma for v in values]
        
        pareto_archs = []
        fight_cache = {}
        for round in range(n_rounds):
            architectures = list(pareto_archs)
            while len(architectures) < n_archs:
                architectures.append(generator.generate(self.architecture_size))
            pareto_archs = []
            learnability_scores = [0 for _ in range(n_archs)]
            simplicity_scores = [0 for _ in range(n_archs)]
            
            n_fights = 0
            total_pairs =  n_archs * (n_archs - 1) // 2
            round_score_cache = {}
            for i in range(n_archs):
                for j in range(i + 1, n_archs):

                    if verbose:
                        print(f"Fight {n_fights + 1}/{total_pairs}: arch {i} vs {j}")

                    if (i,j) in fight_cache:
                        score_i, score_j = fight_cache[(i,j)]
                    elif (j, i) in fight_cache: 
                        score_j, score_i = fight_cache[(j, i)]
                    else:
                        score_i, score_j = self.get_scores(
                        architectures[i], architectures[j], randomizeHP=randomizeHP, pcp=0
                        )
                    round_score_cache[(i,j)] = (score_i, score_j)
                    learnability_scores[i] += math.log((max(score_i,1e-10)))
                    learnability_scores[j] += math.log((max(score_j,1e-10)))
                    simplicity_scores[i] += math.log((max(score_j,1e-10)))
                    simplicity_scores[j] += math.log((max(score_i,1e-10)))
                    if verbose:
                        print(f"score_{i} : {score_i}, score_{j} : {score_j}")
                    
                    n_fights += 1
            
            
            # identifying the pareto front
            pareto_archs_idx = []
            for i in range(n_archs):
                is_dominated = False
                for j in range(n_archs):
                    if learnability_scores[i] < learnability_scores[j] and simplicity_scores[i] < simplicity_scores[j]:
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_archs.append(architectures[i])
                    pareto_archs_idx.append(i)
                    
            # if no arch is dominated, we pop the arch with the worst score
            if len(pareto_archs) == n_archs:
                norm_learn = z_normalize(learnability_scores)
                norm_simp = z_normalize(simplicity_scores)
                occam_scores = [((norm_learn[i]*(1-simp_bal)) + (norm_simp[i]*(simp_bal))) for i in range(n_archs)]
                min_score_idx = occam_scores.index(min(occam_scores))
                # get the index of the min score in the pareto list using pareto_archs_idx
                min_score_idx = pareto_archs_idx.index(min_score_idx)
                pareto_archs.pop(min_score_idx)
                pareto_archs_idx.pop(min_score_idx)
                

            
            #build the cache
            fight_cache = {}
            for i in range(len(pareto_archs)):
                for j in range(i + 1, len(pareto_archs)):
                    fight_cache[(i,j)] = round_score_cache[(pareto_archs_idx[i], pareto_archs_idx[j])]

            
        
        
        learnability_scores = [score/(n_archs-1) for score in learnability_scores]
        simplicity_scores = [score/(n_archs-1) for score in simplicity_scores]
        norm_learn = z_normalize(learnability_scores)
        norm_simp = z_normalize(simplicity_scores)
        occam_scores = [((norm_learn[i]*(1-simp_bal)) + (norm_simp[i]*(simp_bal)))
                         for i in range(n_archs)]
        max_score_idx = occam_scores.index(max(occam_scores))

        if verbose:
            print(f"average learnability score: {sum(norm_learn)/len(norm_learn)}")
            print(f"average simplicity score: {sum(norm_simp)/len(norm_simp)}")


        return architectures[max_score_idx], occam_scores, max_score_idx, learnability_scores, simplicity_scores
        
    
    def tune_speed_bal(self, n_archs=12, n_rounds=4, verbose=False, randomizeHP=True, use_MLPs=True):     
        generator = Generator(generation_type=self.generation_type)
      
        speed_bal_values = []
        for round in range(n_rounds):
            if use_MLPs:
                dims = [[8], [16], [8,16], [16,32], [16,16,32],[16,32,64]]
                architectures = [self.make_mlp(hidden_sizes=size) for size in dims]
                while len(architectures) < n_archs:
                    architectures.append(generator.generate(self.architecture_size))
            n_pairs = n_archs * (n_archs - 1) // 2
            learnability_scores = [0 for _ in range(n_archs)]
            speed_scores = [0 for _ in range(n_archs)]
            n_fight = 0
            for i in range(n_archs):
                for j in range(i + 1, n_archs):
                    if verbose:
                        print(f"fight n°{n_fight+1}/{n_pairs}: arch {i} vs {j}")
                    n_fight +=1
                    score_i, score_j, speed_i, speed_j = self.get_scores(
                        architectures[i], architectures[j], randomizeHP=randomizeHP, get_delays=True
                    )
            
                    learnability_scores[i] += math.log((max(score_i,1e-10)))
                    learnability_scores[j] += math.log((max(score_j,1e-10)))
                    speed_scores[i] += math.log((1/max(speed_i,1e-10)))
                    speed_scores[j] += math.log(1/(max(speed_j,1e-10)))
                    if verbose:
                        print(f"score_{i} : {score_i}, score_{j} : {score_j}")
            learnability_scores = [score/(n_archs-1) for score in learnability_scores]
            speed_scores = [score/(n_archs-1) for score in speed_scores]
            norm_learn = [(learn-self.avg_learn)/self.std_learn for learn in learnability_scores]
            norm_speed = [(speed-self.avg_speed)/self.std_speed for speed in speed_scores]
            
            # Now we find the speed_bal value that minmizes the distances between the occam_scores of the mlp
            norm_learn_mlp = norm_learn[:len(dims)]
            norm_speed_mlp = norm_speed[:len(dims)]


            # Now we find the simp_bal value that minmizes the distances between the occam_scores of the mlp
            norm_learn_mlp = norm_learn[:len(dims)]
            norm_speed_mlp = norm_speed[:len(dims)]
            
            if verbose:
                print(f"norm_learn_mlp: {norm_learn_mlp}")
                print(f"norm_speed_mlp: {norm_speed_mlp}")

            mu_L = sum(norm_learn_mlp) / len(norm_learn_mlp)
            mu_S = sum(norm_speed_mlp) / len(norm_speed_mlp)
            
            num = 0.0
            den = 0.0
            for L, S in zip(norm_learn_mlp, norm_speed_mlp):
                delta_L = L - mu_L
                delta_D = (S - mu_S) - (L - mu_L)
                
                num += delta_L * delta_D
                den += delta_D * delta_D
                
            if den == 0:
                best_speed_bal = 0.5 # Fallback if all scores are identical
            else:
                best_speed_bal = - (num / den)
                
            # Clip between 0 and 1 to keep it a valid percentage
            best_speed_bal = max(0.0, min(1.0, best_speed_bal))
            
            if verbose:
                print(f"Round {round} optimal speed_bal: {best_speed_bal:.4f}")
            speed_bal_values.append(best_speed_bal)

        avg_speed_bal = sum(speed_bal_values) / len(speed_bal_values)
        std_speed_bal = math.sqrt(sum((v - avg_speed_bal) ** 2 for v in speed_bal_values) / len(speed_bal_values))
        return speed_bal_values , avg_speed_bal, std_speed_bal

    def get_learnability_and_delays_data(self, n_tests=256, recent_size=16, verbose=False, randomizeHP=True, simp_bal=None):
        if simp_bal is None:
            simp_bal = self.simp_bal
        generator = Generator(generation_type=self.generation_type)
        learnabilities = []
        speeds = []
        recent_learn_means = []
        recent_learn_stds = []
        recent_speed_means = []
        recent_speed_stds = []

        # Prefill
        correct_generation = False
        while correct_generation==False:
            arch_A = generator.generate(self.architecture_size)
            arch_B = generator.generate(self.architecture_size)
            score_A, score_B, delay_A, delay_B = self.get_scores(arch_A, arch_B, randomizeHP=randomizeHP, pcp=0, get_delays=True)
            if self._valid(score_A, score_B, delay_A, delay_B):
                correct_generation = True
            else:
                if verbose:
                    print(f"Scores are too low, trying again")
                continue
            learnabilities.append(math.log(max(score_A, 1e-10)))
            learnabilities.append(math.log(max(score_B, 1e-10)))
            speeds.append(math.log(max(1.0 / max(delay_A, 1e-10), 1e-10)))
            speeds.append(math.log(max(1.0 / max(delay_B, 1e-10), 1e-10)))

        learn_means_std = 0
        learn_stds_std = 0
        speed_means_std = 0
        speed_stds_std = 0

        for test in range(n_tests):
            if verbose:
                print(f"Test {test+1}/{n_tests}")
            correct_generation = False
            while correct_generation==False:
                arch_A = generator.generate(self.architecture_size)
                arch_B = generator.generate(self.architecture_size)
                score_A, score_B, delay_A, delay_B = self.get_scores(arch_A, arch_B, randomizeHP=randomizeHP, pcp=0, get_delays=True)
                if self._valid(score_A, score_B, delay_A, delay_B):
                    correct_generation = True
                else:
                    if verbose:
                        print(f"Scores are too low, trying again")
                    continue
                learnabilities.append(math.log(max(score_A, 1e-10)))
                learnabilities.append(math.log(max(score_B, 1e-10)))
                speeds.append(math.log(max(1.0 / max(delay_A, 1e-10), 1e-10)))
                speeds.append(math.log(max(1.0 / max(delay_B, 1e-10), 1e-10)))

            # Learnability stats
            learn_mean = sum(learnabilities) / len(learnabilities)
            learn_std = math.sqrt(sum((v - learn_mean) ** 2 for v in learnabilities) / len(learnabilities))
            recent_learn_means.append(learn_mean)
            recent_learn_stds.append(learn_std)

            # Speed stats
            speed_mean = sum(speeds) / len(speeds)
            speed_std = math.sqrt(sum((v - speed_mean) ** 2 for v in speeds) / len(speeds))
            recent_speed_means.append(speed_mean)
            recent_speed_stds.append(speed_std)

            if len(recent_learn_means) > recent_size:
                recent_learn_means.pop(0)
                recent_learn_stds.pop(0)
                recent_speed_means.pop(0)
                recent_speed_stds.pop(0)

                learn_means_std = math.sqrt(sum((v - recent_learn_means[-1]) ** 2 for v in recent_learn_means) / len(recent_learn_means))
                learn_stds_std = math.sqrt(sum((v - recent_learn_stds[-1]) ** 2 for v in recent_learn_stds) / len(recent_learn_stds))
                speed_means_std = math.sqrt(sum((v - recent_speed_means[-1]) ** 2 for v in recent_speed_means) / len(recent_speed_means))
                speed_stds_std = math.sqrt(sum((v - recent_speed_stds[-1]) ** 2 for v in recent_speed_stds) / len(recent_speed_stds))

                if verbose:
                    print(f"  Learn convergence: mean_std={learn_means_std:.4f}, std_std={learn_stds_std:.4f}")
                    print(f"  Speed convergence: mean_std={speed_means_std:.4f}, std_std={speed_stds_std:.4f}")

            print(f"Learn mean: {learn_mean:.4f}, std: {learn_std:.4f} | Speed mean: {speed_mean:.4f}, std: {speed_std:.4f}")

        print(f"\n=== Final Results ===")
        print(f"Learnability: mean={learn_mean:.4f}, std={learn_std:.4f}")
        print(f"Speed:        mean={speed_mean:.4f}, std={speed_std:.4f}")
        print(f"Learn convergence: mean_std={learn_means_std:.4f}, std_std={learn_stds_std:.4f}")
        print(f"Speed convergence: mean_std={speed_means_std:.4f}, std_std={speed_stds_std:.4f}")

        return learn_mean, learn_std, speed_mean, speed_std
                
    def fast_find_golden_pool(self, n_pools=10, pool_size=11, n_refs=4, n_refs_tests=2, n_tst_pools=6, verbose=False, randomizeHP=True, simp_bal=None):
        if simp_bal is None:
            simp_bal = self.simp_bal
        generator = Generator(generation_type=self.generation_type)
        references_bank = [generator.generate(self.architecture_size) for _ in range(n_refs)]
        pool_bank = []
        if verbose:
            print(f"Generating {n_pools} pools of {pool_size} architectures...")
        for i in range(n_pools):
            pool_bank.append([generator.generate(self.architecture_size) for _ in range(pool_size)])
        
        ref_occams = {}
        for ref in range(len(references_bank)):
            ref_occams[f'ref_{ref}'] = []
        
        
        for i_pool, pool in enumerate(pool_bank):
            archs=[]
            if verbose:
                print(f"Testing pool {i_pool+1}/{n_pools}")
            archs = references_bank+pool
            wrs, occam_scores, norm_learn, norm_simp = self.occam_test(archs,n_archs=len(archs), use_delays=True)
            for ref in range(len(references_bank)):
                ref_occams[f'ref_{ref}'].append(occam_scores[ref])
        # find the pool with the lowest average distance to the mean occam score across refs
        pool_errors = []
        averages = {}
        for ref in range(len(references_bank)):
            averages[f'ref_{ref}'] = sum(ref_occams[f'ref_{ref}']) / len(ref_occams[f'ref_{ref}'])
        deltas = {}
        for ref in range(len(references_bank)):
            for occam in ref_occams[f'ref_{ref}']:
                delta = abs(occam - averages[f'ref_{ref}'])
                if f'ref_{ref}' not in deltas:
                    deltas[f'ref_{ref}'] = []
                deltas[f'ref_{ref}'].append(delta)
        
        pools_errors = []
        for i_pool in range(n_pools):
            pool_error = 0
            for ref in range(len(references_bank)):
                pool_error += deltas[f'ref_{ref}'][i_pool]
            pools_errors.append(pool_error)
        
        min_error_idx = pools_errors.index(min(pools_errors))
        golden_pool = pool_bank[min_error_idx]
        # save the golden pool
        save_dir = "golden_pool_archs"
        os.makedirs(save_dir, exist_ok=True)
        for i, arch in enumerate(golden_pool):
            filename = os.path.join(save_dir, f"golden_arch_{i}.pkl")
            arch.save(filename)
            if verbose:
                print(f"Saved {filename}")
        if verbose:
            print("Starting Evalutation of Golden Pool...")
        golden_pool_errors = []
        for ref_test in range(n_refs_tests):
            ref_arch = generator.generate(self.architecture_size)
            archs = [ref_arch] + golden_pool
            wrs, occam_scores, norm_learn, norm_simp = self.occam_test(archs,n_archs=len(archs), use_delays=True)
            golden_occam = occam_scores[0]

            random_pools_occams = []
            for tst in range(n_tst_pools):
                random_pool = [generator.generate(self.architecture_size) for _ in range(pool_size)]
                archs = [ref_arch] + random_pool
                wrs, occam_scores, norm_learn, norm_simp = self.occam_test(archs,n_archs=len(archs), use_delays=True)
                random_pools_occams.append(occam_scores[0])
            mean_random_occam = sum(random_pools_occams) / len(random_pools_occams)
            golden_pool_errors.append(abs(golden_occam - mean_random_occam))  

        avg_golden_pool_error = sum(golden_pool_errors) / len(golden_pool_errors)
        avg_random_pool_error = sum([abs(occam - mean_random_occam) for occam in random_pools_occams]) / len(random_pools_occams)
        if verbose:
            print(f"Average absolute distance of Golden Pool from True Mean on UNSEEN archs: {avg_golden_pool_error:.4f}")
            print(f"Average absolute distance of Random Pools from True Mean on UNSEEN archs: {avg_random_pool_error:.4f}")
        
        return golden_pool, avg_golden_pool_error, avg_random_pool_error
            
            
    
    def make_mlp(self,hidden_sizes, inputTens=None):
        if inputTens is None:
            input_f = hidden_sizes[0]
        else :
            input_f = inputTens.shape[-1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MLP_arch = Architecture()
        inputModule = Input()
        MLP_arch.add_node(0,inputModule)

        prev_f = input_f
        last_output_node = 0

        for i, hidden_f in enumerate(hidden_sizes):                   
            weight = MLP_arch.append_node(LearnableParameter((1,prev_f,hidden_f)))
            bias = MLP_arch.append_node(LearnableParameter((1,1,hidden_f)))
            matmul = MLP_arch.append_node(MatMul())
            addition = MLP_arch.append_node(Add())
            if i != len(hidden_sizes)-1:
                activation = MLP_arch.append_node(Activation())

            #edges
            MLP_arch.add_edge(last_output_node,matmul)
            MLP_arch.add_edge(weight,matmul)
            MLP_arch.add_edge(matmul,addition)
            MLP_arch.add_edge(bias,addition)
            if i != len(hidden_sizes)-1:
                MLP_arch.add_edge(addition,activation)
                last_output_node = activation
            else:
                last_output_node = addition # useless in theory but might have a use in the future
            prev_f = hidden_f
        return MLP_arch


    def _load_real_datasets(self):
        """Load and cache all real datasets once."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        datasets = {}

        # California
        data = fetch_california_housing()
        X, y = data.data, data.target
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        datasets['california_housing'] = {
            'train_input': torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
            'train_target': torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1, 1),
            'test_input': torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
            'test_target': torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1, 1),
        }

        # MNIST
        train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

        datasets['mnist'] = {
            'train_input': train_ds.data.float() / 255.0,
            'train_target': torch.nn.functional.one_hot(train_ds.targets, 10).float().unsqueeze(1),
            'test_input': test_ds.data.float() / 255.0,
            'test_target': torch.nn.functional.one_hot(test_ds.targets, 10).float().unsqueeze(1),
        }

        # CIFAR-10
        train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

        train_imgs = torch.stack([img for img, _ in train_ds])
        test_imgs = torch.stack([img for img, _ in test_ds])

        datasets['cifar10'] = {
            'train_input': train_imgs.permute(0, 2, 1, 3).contiguous().view(-1, 32, 96),
            'train_target': torch.nn.functional.one_hot(torch.tensor([l for _, l in train_ds]), 10).float().unsqueeze(1),
            'test_input': test_imgs.permute(0, 2, 1, 3).contiguous().view(-1, 32, 96),
            'test_target': torch.nn.functional.one_hot(torch.tensor([l for _, l in test_ds]), 10).float().unsqueeze(1),
        }

        return datasets
    

    def realDataSet_test_cached(self, architecture, datasets, verbose=True, max_iter=100, subsample=5000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        max_batch_size = 2048 if device.type == 'cuda' else 32
        results = {}

        def batched_forward(executor, data, batch_size=2048):
            outputs = []
            with torch.no_grad():
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size].to(device)
                    out = executor.forward(batch)
                    outputs.append(out[0].cpu())
            return [torch.cat(outputs, dim=0).to(device)]

        for name, ds in datasets.items():
            try:
                train_input = ds['train_input']
                train_target = ds['train_target']
                test_input = ds['test_input']
                test_target = ds['test_target']

                # Subsample training data
                if subsample and len(train_input) > subsample:
                    idx = torch.randperm(len(train_input))[:subsample]
                    train_input = train_input[idx]
                    train_target = train_target[idx]

                train_input = train_input.to(device)
                train_target = train_target.to(device)
                test_input = test_input.to(device)
                test_target = test_target.to(device)

                arch_copy = copy.deepcopy(architecture)
                arch_copy.reset_state()
                executor = Executor(arch_copy).to(device)

                executor.randomize_weights()

                start_fit = time.time()
                executor.fit(
                    train_input, train_target,
                    verbose=verbose, lr=0.001, max_iter=max_iter,
                    batch_size=min(len(train_input), max_batch_size),
                    patience=10, min_delta=1e-7, cpu=False
                )
                fit_delay = time.time() - start_fit

                start_test = time.time()
                test_output = batched_forward(executor, test_input)
                test_delay = time.time() - start_test

                loss_val = torch.nn.functional.mse_loss(test_output[0], test_target).item()

                if not math.isfinite(loss_val) or loss_val > 1e6:
                    loss_val = float('nan')
                    test_score = 0.0

                else :
                    test_score = 1.0 / (loss_val + 1e-8)
                results[name] = {
                    'test_score': test_score,
                    'fit_delay': fit_delay,
                    'test_delay': test_delay
                }

                del executor
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  {name} failed for architecture: {e}")
                results[name] = {
                    'test_loss': float('nan'),
                    'fit_delay': float('nan'),
                    'test_delay': float('nan')
                }
                torch.cuda.empty_cache()

        return results



    def get_distinction(self,architecture, n_archs=32, verbose = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type=='cuda':
            max_batch_size = 2048
        else:
            max_batch_size = 32
        arch_scores = []
        contestant_scores = []
        generator = Generator(generation_type=self.generation_type)

        for n_fight in range(n_archs):
            if verbose:
                print(f"Fight n°{n_fight+1}/{n_archs}")
            contestant = generator.generate(self.architecture_size)
            arch_score, contestant_score = self.get_scores(architecture, contestant)
            arch_scores.append(arch_score)
            contestant_scores.append(contestant_score)
        return arch_scores, contestant_scores


    
    def test_real_correlation(self,architectures,n_archs_test=16,simp_bal=None, real_iter = 150, verbose = True, save_path="correlation_data.csv", use_golden_pool=False):
        # We evaluate every architecture on both the arena and the real data set
        # We compute the correlation between the architectures' scores on the real data set and the scores on the arena
        if simp_bal is None:
            simp_bal = self.simp_bal
        ############################ Arena metrics ############################
        learnabilities = []
        simplicities = []
        occam_scores = []
        generator = Generator(generation_type=self.generation_type)
        fixed_opponents = [generator.generate(self.architecture_size) for _ in range(n_archs_test - 1)]
        n_opp = len(fixed_opponents)

        # Precompute opponent-vs-opponent scores
        # opp_log_scores[i][j] = log(score of opponent i when learning opponent j)
        # opp_log_scores[j][i] = log(score of opponent j when learning opponent i)
        opp_log_scores = [[0.0] * n_opp for _ in range(n_opp)]
        for a in range(n_opp):
            if verbose:
                print(f"Opponent {a+1}/{n_opp}")
            for b in range(a + 1, n_opp):
                score_a, score_b = self.get_scores(fixed_opponents[a], fixed_opponents[b],randomizeHP=False, pcp=0)
                opp_log_scores[a][b] = math.log(max(score_a, 1e-10))
                opp_log_scores[b][a] = math.log(max(score_b, 1e-10))
        
        if verbose:
            print("Opponent cache built.")

        # Z-normalize, used later but defined now to avoid recomputing it
        def z_normalize(values):
            mu = sum(values) / len(values)
            var = sum((v - mu) ** 2 for v in values) / len(values)
            sigma = math.sqrt(var) if var > 0 else 1.0
            return [(v - mu) / sigma for v in values]

        for i, arch in enumerate(architectures):
            if verbose:
                print(f"arena-Testing architecture {i+1}/{len(architectures)}")
            tested_learn_scores = []
            tested_simp_scores = []
            for j in range(n_opp):
                score_tested, score_opp = self.get_scores(arch, fixed_opponents[j], randomizeHP=False, pcp=0)
                tested_learn_scores.append(math.log(max(score_tested, 1e-10)))
                tested_simp_scores.append(math.log(max(score_opp, 1e-10)))

            pool_size = n_opp + 1  # should equal n_archs_test

            raw_learn = [0.0] * pool_size
            raw_simp = [0.0] * pool_size

            for j in range(n_opp):
                raw_learn[0] += tested_learn_scores[j]       # tested learned opp_j
                raw_simp[0] += tested_simp_scores[j]         # opp_j learned tested
                raw_learn[j + 1] += tested_simp_scores[j]    # opp_j learned tested
                raw_simp[j + 1] += tested_learn_scores[j]    # tested learned opp_j

            # Opponent vs opponent (cached)
            for a in range(n_opp):
                for b in range(a + 1, n_opp):
                    raw_learn[a + 1] += opp_log_scores[a][b]
                    raw_learn[b + 1] += opp_log_scores[b][a]
                    raw_simp[a + 1] += opp_log_scores[b][a]
                    raw_simp[b + 1] += opp_log_scores[a][b]

            # Normalize by number of opponents
            raw_learn = [s / (pool_size - 1) for s in raw_learn]
            raw_simp = [s / (pool_size - 1) for s in raw_simp]

            

            norm_learn = z_normalize(raw_learn)
            norm_simp = z_normalize(raw_simp)

            occam = [(1 - simp_bal) * norm_learn[k] + simp_bal * norm_simp[k] for k in range(pool_size)]

            learnabilities.append(norm_learn[0])
            simplicities.append(norm_simp[0])
            occam_scores.append(occam[0])
                

        arena_metrics = {"learnability": learnabilities, "simplicity": simplicities, "occam_score": occam_scores}
        ############################ Real data set metrics ############################


        real_dataset_metrics = defaultdict(list)# dict of ["dataset-metrics"]->list of values
        datasets = self._load_real_datasets()
        for i, arch in enumerate(architectures):
            if verbose:
                print(f"real-Testing architecture {i}")
            
            res = self.realDataSet_test_cached(arch, datasets, verbose=False, max_iter=real_iter)
            for dataset in res.keys():
                for metric in res[dataset].keys():
                    real_dataset_metrics[f"{dataset}-{metric}"].append(res[dataset][metric])

        
        ########################### Compute correlations ##############################
        if verbose:
            print(f"Computing correlations...")
        correlations = {} # dict of ["arenametric-dataset-realdatametric"]->correlation
        for arenametric in arena_metrics:
            for realdatametric in real_dataset_metrics:
                correlations[f"{arenametric}-{realdatametric}"] = spearmanr(arena_metrics[arenametric], real_dataset_metrics[realdatametric])[0]
        if verbose:
            print(correlations)
        
        
        ########################## Save to CSV ##############################²
        headers = ["arch_id", "learnability", "simplicity", "occam_score"]
        headers += sorted(real_dataset_metrics.keys())

        rows = []

        for i in range(len(architectures)):
            row = {
                "arch_id": i,
                "learnability": learnabilities[i],
                "simplicity": simplicities[i],
                "occam_score": occam_scores[i],
            }
            for key in sorted(real_dataset_metrics.keys()):
                row[key] = real_dataset_metrics[key][i]
            rows.append(row)
        
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"Saved {len(rows)} rows to {save_path}")


        
        return correlations
    
    def corr_data_processing(self, save_path="correlation_data.csv"):
        rows = []
        with open(save_path, "r") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v)
                    except (ValueError, TypeError):
                        parsed[k] = float('nan')
                rows.append(parsed)
        # rows are dicts
        # we want to add a column for average z-normalized score on real test set
        def z_normalize(values):
            mu = sum(values) / len(values)
            var = sum((v - mu) ** 2 for v in values) / len(values)
            sigma = math.sqrt(var) if var > 0 else 1.0
            return [(v - mu) / sigma for v in values]
        
        # Z-normalize each dataset's test losses separately, then average
        loss_keys = [k for k in headers if k.endswith("-test_score")]
        per_dataset_z = {}
        for lk in loss_keys:
            vals = [r[lk] for r in rows]
            per_dataset_z[lk] = z_normalize(vals)
        
        for i in range(len(rows)):
            avg_z = sum(per_dataset_z[lk][i] for lk in loss_keys) / len(loss_keys)
            rows[i]["avg_z_test_score"] = avg_z
        
        # Compute correlations
        arena_keys = ["learnability", "simplicity", "occam_score"]
        avg_z_vals = [r["avg_z_test_score"] for r in rows]
        
        correlations = {}
        for ak in arena_keys:
            a_vals = [r[ak] for r in rows]
            sp = spearmanr(a_vals, avg_z_vals)[0]
            correlations[ak] = {"spearman": sp}
        
        os.makedirs("plots", exist_ok=True)
        for ak in arena_keys:
            a_vals = [r[ak] for r in rows]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(a_vals, avg_z_vals, s=60, edgecolors='k', linewidths=0.5)
            for j, (x, y) in enumerate(zip(a_vals, avg_z_vals)):
                ax.annotate(str(j), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
            sp = correlations[ak]["spearman"]
            ax.set_title(f"{ak} vs Avg Z-Normalized Test Score (Spearman={sp:.3f})")
            ax.set_xlabel(ak)
            ax.set_ylabel("Avg Z-Normalized Test Score")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"plots/{ak}_vs_avg_z_test.png", dpi=150)
            plt.close(fig)

        # Append correlations to CSV
        with open(save_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["# Correlations vs avg_z_test_score"])
            writer.writerow(["arena_metric", "spearman"])
            for ak, v in correlations.items():
                writer.writerow([ak, f"{v['spearman']:.4f}"])

        # Plot simplicity vs avg z-fit delay
        fit_keys = [k for k in headers if k.endswith("-fit_delay")]

        #  Z-normalize each dataset's fit delays separately
        per_dataset_z_fit = {}
        for fk in fit_keys:
            vals = [r[fk] for r in rows]
            per_dataset_z_fit[fk] = z_normalize(vals)

        # Average the normalized fit delays and add to rows
        for i in range(len(rows)):
            avg_z_f = sum(per_dataset_z_fit[fk][i] for fk in fit_keys) / len(fit_keys)
            rows[i]["avg_z_fit"] = avg_z_f
        
        s_vals, f_vals = [r["simplicity"] for r in rows], [r["avg_z_fit"] for r in rows]
        sp_fit = spearmanr(s_vals, f_vals)[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(s_vals, f_vals, c='g')
        for j, (x, y) in enumerate(zip(s_vals, f_vals)): ax.annotate(str(j), (x,y), xytext=(3,3), textcoords="offset points")
        ax.set_title(f"Simplicity vs Avg Z-Fit Delay (Sp={sp_fit:.3f})")
        fig.savefig("plots/simplicity_vs_z_fit_delay.png", bbox_inches='tight'); plt.close(fig)
        
                    
                    


        
        

        