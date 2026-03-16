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


from modules.base import ModuleType
from modules.operations import *
from modules.learnable import *
from modules.input import *
from graph.architecture import *
from graph.executor import *
from graph.generator import *

class Arena:
    def __init__(self, n_fights=1, architecture_size=16, arena_contestants=3, dataset_size = 256+64, train_test_split= 0.7, generation_type="agnostic", verbose=True, report=False, pcp=0.38, cpu=False):
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

    def calibrate_pcp(self, n_fights=128, min_nodes=4, max_nodes=24,
                    initial_step=0.1, step_decay=0.98, verbose=True, 
                    finalvalsize=32):
        """
        Fights architectures of different sizes against each other.
        Adjusts PCP so that size alone doesn't predict the winner.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = Generator(generation_type=self.generation_type)
        step = initial_step
        
        best_outerfunction = "log2"
        best_distance = float('inf')
        best_pcp = self.pcp
        for outerfunction in ["sqrt"]:
            self.pcp = 0.38  # reset for each outer function
            step = initial_step
            
            for fight in range(n_fights):
                if fight % 5 == 0:
                    print(f"Fight {fight}/{n_fights} for {outerfunction}, current PCP: {self.pcp:.4f}")
                # Keep sizes reasonable to avoid OOM
                size1 = random.randint(min_nodes, max_nodes)
                size2 = random.randint(min_nodes, max_nodes)
                
                while abs(size1 - size2) < 2:
                    size2 = random.randint(min_nodes, max_nodes)
                
                firstisbigger = size1 > size2
                
                try:
                    arch1 = generator.generate(size1)
                    arch2 = generator.generate(size2)

                    firstisbigger = arch1.parameter_count() > arch2.parameter_count()
                    score1, score2 = self.get_scores(arch1, arch2, 
                                                    outerfunction=outerfunction)
                    firstwon = score1 > score2
                    
                    # Size-proportional adjustment
                    size_ratio = max(arch1.parameter_count(), arch2.parameter_count()) / max(1, min(arch1.parameter_count(), arch2.parameter_count()))
                    adjustment = step * math.log2(size_ratio)
                    
                    if firstwon == firstisbigger:
                        # Bigger won → increase penalty
                        self.pcp += adjustment
                    else:
                        # Smaller won → decrease penalty
                        self.pcp -= adjustment
                    
                    self.pcp = max(0.0, self.pcp)
                    step *= step_decay
                    
                except (RuntimeError, Exception) as e:
                    if "out of memory" in str(e).lower():
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        if verbose:
                            print(f"  OOM on sizes {size1},{size2} — skipping")
                        continue
                    else:
                        if verbose:
                            print(f"  Error on sizes {size1},{size2}: {e} — skipping")
                        continue
                finally:
                    # Always clean up
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                if verbose and fight % 10 == 0:
                    print(f"  {outerfunction} fight {fight}/{n_fights} | "
                        f"PCP: {self.pcp:.4f} | step: {step:.4f} | "
                        f"sizes: {size1} vs {size2}")
            
            # Validation
            bigger_wins = 0
            valid_fights = 0
            
            for fight in range(finalvalsize):
                size1 = random.randint(min_nodes, max_nodes)
                size2 = random.randint(min_nodes, max_nodes)
                
                while abs(size1 - size2) < 2:
                    size2 = random.randint(min_nodes, max_nodes)
                
                try:
                    arch1 = generator.generate(size1)
                    arch2 = generator.generate(size2)
                    
                    # Bigger always first
                    if arch2.parameter_count() > arch1.parameter_count():
                        arch1, arch2 = arch2, arch1
                    
                    score1, score2 = self.get_scores(arch1, arch2, 
                                                    outerfunction=outerfunction)
                    valid_fights += 1
                    if score1 > score2:
                        bigger_wins += 1
                        
                except (RuntimeError, Exception) as e:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                finally:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            if valid_fights > 0:
                distance = abs(bigger_wins / valid_fights - 0.5)
                print(f"\n{outerfunction}: PCP={self.pcp:.4f} | "
                    f"bigger won {bigger_wins}/{valid_fights} | "
                    f"distance to 0.5: {distance:.4f}")
                
                if distance < best_distance:
                    best_distance = distance
                    best_outerfunction = outerfunction
                    best_pcp = self.pcp
        
        self.pcp = best_pcp
        self.outerfunction = best_outerfunction
        print(f"\nBest: {best_outerfunction} with PCP={best_pcp:.4f} "
            f"(distance={best_distance:.4f})")
        return self.pcp, best_outerfunction

    def get_scores(self, arch_1, arch_2, input = None, get_penalties=False, outerfunction="sqrt", randomizeHP=False, pcp=None):
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
            input = torch.randn(self.dataset_size, input_p, input_f).to(device)

        train_size = int(self.dataset_size*train_test_split)
        train_input = input[:train_size]
        test_input = input[train_size:]

        # generate the outputs for each executor
        output_1 = executors[0].forward(input)
        output_2 = executors[1].forward(input)

        train_target_1 = output_2[0][:train_size].to(device)
        train_target_2 = output_1[0][:train_size].to(device)

        test_target_1 = output_2[0][train_size:].to(device)
        test_target_2 = output_1[0][train_size:].to(device)

        # make new executors and fit them
        learner_1 = Executor(copy.deepcopy(arch_1)).to(device)
        learner_2 = Executor(copy.deepcopy(arch_2)).to(device)
        if randomizeHP:
            learner_1.fit(train_input, train_target_1.detach(), verbose=self.verbose, lr=random_lr, max_iter=random_max_iter, batch_size=min(train_size,random_batch_size), patience = random_patience, min_delta = random_min_delta, cpu = False)
            learner_2.fit(train_input, train_target_2.detach(), verbose=self.verbose, lr=random_lr, max_iter=random_max_iter, batch_size=min(train_size,random_batch_size), patience = random_patience, min_delta = random_min_delta, cpu = False)
        else:
            learner_1.fit(train_input, train_target_1.detach(), verbose=self.verbose, lr=0.01, max_iter=200, batch_size=min(train_size,max_batch_size), patience = 10, min_delta = 1e-7, cpu = False)
            learner_2.fit(train_input, train_target_2.detach(), verbose=self.verbose, lr=0.01, max_iter=200, batch_size=min(train_size,max_batch_size), patience = 10, min_delta = 1e-7, cpu = False)

        with torch.no_grad():
            pred_1 = learner_1.forward(test_input)[0]
            pred_2 = learner_2.forward(test_input)[0]
            
            test_loss_1 = torch.nn.functional.mse_loss(pred_1, test_target_1).item()
            test_loss_2 = torch.nn.functional.mse_loss(pred_2, test_target_2).item()

            if pcp==0:
                score_1 = (1/(test_loss_1))**0.5 # the higher the better
                score_2 = (1/(test_loss_2))**0.5
                del executors[0]
                del executors[0], learner_1, learner_2 #because executors[0] is our previous executors[1] (executors[0] was deleted, which made executors[1] become executors[0]))
                torch.cuda.empty_cache()
                return score_1, score_2

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
            return score_1, score_2

    def occam_selection(self, n_archs=16, verbose=False, randomizeHP=True, simp_bal=0.3):
        generator = Generator(generation_type=self.generation_type)
        architectures = [generator.generate(self.architecture_size) for _ in range(n_archs)]
        simplicity_scores = [0 for _ in range(n_archs)]
        learnability_scores = [0 for _ in range(n_archs)]
        
        
        n_fights = 0
        total_pairs =  n_archs * (n_archs - 1) // 2
        for i in range(n_archs):
            for j in range(i + 1, n_archs):
                if verbose:
                    print(f"Fight {n_fights + 1}/{total_pairs}: arch {i} vs {j}")
                
                score_i, score_j = self.get_scores(
                    architectures[i], architectures[j], randomizeHP=randomizeHP, pcp=0
                )
                learnability_scores[i] += math.log((max(score_i,1e-10)))
                learnability_scores[j] += math.log((max(score_j,1e-10)))
                simplicity_scores[i] += math.log((max(score_j,1e-10)))
                simplicity_scores[j] += math.log((max(score_i,1e-10)))
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
        norm_learn = z_normalize(learnability_scores)
        norm_simp = z_normalize(simplicity_scores)
        occam_scores = [((norm_learn[i]*(1-simp_bal)) + (norm_simp[i]*(simp_bal)))
                         for i in range(n_archs)]
        max_score_idx = occam_scores.index(max(occam_scores))

        if verbose:
            print(f"average learnability score: {sum(norm_learn)/len(norm_learn)}")
            print(f"average simplicity score: {sum(norm_simp)/len(norm_simp)}")


        return architectures[max_score_idx], occam_scores, max_score_idx, learnability_scores, simplicity_scores
        
    def occam_test(self, ori_architectures, n_archs=8, verbose=False, randomizeHP=True, simp_bal=0.3):

        if not isinstance(ori_architectures, list):
            ori_architectures = [ori_architectures]
        
        generator = Generator(generation_type=self.generation_type)
        architectures = copy.deepcopy(ori_architectures)

        for i in range(n_archs-len(architectures)):
            architectures.append(generator.generate(self.architecture_size))

        learnabilities = [0 for _ in range(n_archs)]
        simplicities = [0 for _ in range(n_archs)]
        
        n_fight = 0
        total_pairs =  n_archs * (n_archs - 1) // 2
        for i in range(n_archs):
            for j in range(i + 1, n_archs):
                if verbose:
                    print(f"Fight {n_fight + 1}/{total_pairs}: arch {i} vs {j}")
                score_i, score_j = self.get_scores(
                    architectures[i], architectures[j], randomizeHP=randomizeHP, pcp=0
                )

                learnabilities[i] += math.log((max(score_i,1e-10)))
                learnabilities[j] += math.log((max(score_j,1e-10)))
                simplicities[i] += math.log((max(score_j,1e-10)))
                simplicities[j] += math.log((max(score_i,1e-10)))
                if verbose:
                    print(f"score_{i} : {math.log((max(score_i,1e-10)))}, score_{j} : {math.log((max(score_j,1e-10)))}")

                n_fight += 1
        learnabilities = [score/(n_archs-1) for score in learnabilities]
        simplicities = [score/(n_archs-1) for score in simplicities]
        occam_scores = [((learnabilities[i]*(1-simp_bal)) + (simplicities[i]*(simp_bal)))
                         for i in range(n_archs)]
        
        occam_scores_sorted = sorted(occam_scores)
        wrs = [0 for _ in range(len(ori_architectures))]
        for tested_arch in range(len(ori_architectures)):
            occam_score = occam_scores[tested_arch]
            rank = occam_scores_sorted.index(occam_score)
            wrs[tested_arch] = rank/(len(occam_scores)-1)

        return wrs, occam_scores, learnabilities, simplicities


    def smooth_selection(self, n_archs=16, max_fights=256, verbose=False, randomizeHP=True):
        generator = Generator(generation_type=self.generation_type)
        architectures = [generator.generate(self.architecture_size) for _ in range(n_archs)]
        arch_scores = [0 for _ in range(n_archs)]
        
        n_fights = 0
        total_pairs = min(max_fights, n_archs * (n_archs - 1) // 2)
        
        for i in range(n_archs):
            for j in range(i + 1, n_archs):
                if verbose:
                    print(f"Fight {n_fights + 1}/{total_pairs}: arch {i} vs {j}")
                
                score_i, score_j = self.get_scores(
                    architectures[i], architectures[j], randomizeHP=randomizeHP
                )
                if score_i > score_j:
                    arch_scores[i] += 1
                else:
                    arch_scores[j] += 1
                
                n_fights += 1
                if n_fights >= max_fights:
                    break
            if n_fights >= max_fights:
                break

        best_arch_index = arch_scores.index(max(arch_scores))
        
        if verbose:
            print(f"Scores: {arch_scores}")
            print(f"Winner: arch {best_arch_index} with {arch_scores[best_arch_index]} wins")
        
        return architectures[best_arch_index], arch_scores, best_arch_index
            
    

    def start(self, randomizeHP = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type=='cuda':
            max_batch_size = 2048
        else:
            max_batch_size = 32
        
        winners = []
        winner_scores = [0]
        current_winner_id = 0

        
            
        # Generate the first architecture
        generator = Generator(generation_type=self.generation_type)
        current_best = generator.generate(self.architecture_size)
        winners.append(copy.deepcopy(current_best))
        for n_fight in range(self.n_fights):
            print(f"fight n°{n_fight}")
            architectures = []
            architectures.append(current_best)
            scores = []
            # arch 1, executor 0
            for gen_architecture in range(self.arena_contestants-1):
                # Generate a new architecture
                new_architecture = generator.generate(self.architecture_size)
                architectures.append(new_architecture)
                
            for architecture in architectures:
                scores.append(0)

            # architectures is filled, now we evaluate them two by two (every possible pair)
            for i in range(self.arena_contestants-1): 
                for j in range(i+1, self.arena_contestants):   
                    score_i, score_j = self.get_scores(architectures[i], architectures[j], randomizeHP=randomizeHP)
                    if score_j < score_i:
                        scores[i] = scores[i] + 1
                    else:
                        scores[j] = scores[j] + 1
                    '''
                    if self.verbose:
                        print(f"Architecture {i} of round {n_fight} :")
                        architectures[i].describe()
                        print(f"Architecture {j} of round {n_fight} :")
                        architectures[j].describe()
                        print(f"Test loss for executor {i}: {test_loss_i}")
                        print(f"Test loss for executor {j}: {test_loss_j}")
                    '''
            # Fight loop
            max_score_id = scores.index(max(scores))
            if max_score_id == 0: # because the first architecture is always the previous best
                winner_scores[current_winner_id] += scores[max_score_id]
            else:
                winners.append(copy.deepcopy(architectures[max_score_id]))
                winner_scores.append(scores[max_score_id])
                current_winner_id += 1
                current_best = copy.deepcopy(architectures[max_score_id])

        if self.verbose:
            print(f"Winners Scores: {winners}")
            print(f"Final scores: {scores}")
            print(f"Final architecture: ")
            current_best.describe()
        return winner_scores, winners
    
    #################### END OF TRAINING ####################

    def test(self,architecture,arena_contestants = 3, n_test=8, verbose=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type=='cuda':
            max_batch_size = 2048
        else:
            max_batch_size = 32
        arch_scores = []
        generator = Generator(generation_type=self.generation_type)
        current_best = architecture
        
        for n_fight in range(n_test):
            if verbose:
                print(f"Test fight n°{n_fight}")
            architectures = []
            architectures.append(current_best)
            scores = []
            
            # arch 1, executor 0
            for gen_architecture in range(arena_contestants-1):
                # Generate a new architecture
                new_architecture = generator.generate(self.architecture_size)
                architectures.append(new_architecture)
                
            for architecture in architectures:
                scores.append(0)

            # architectures is filled, now we evaluate them two by two (every possible pair)
            for i in range(self.arena_contestants-1): 
                for j in range(i+1, self.arena_contestants):
                    score_i, score_j = self.get_scores(architectures[i], architectures[j])
                    if score_i > score_j:
                        scores[i] = scores[i] + 1
                    else:
                        scores[j] = scores[j] + 1
            # Fight loop
            arch_scores.append(scores)

        # We compute the number of times the first architecture was the best
        # print(arch_scores)
        n_wins = 0
        for score in arch_scores:
            if score[0] == max(score):
                n_wins += 1
        return arch_scores, n_wins
            
    
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

    ############################ MLP COMPARISON ############################

    def test_mlp(self, architecture, mlp_n_tests=64, mlp_hidden_sizes=[32,32,16], verbose=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        scores=[]
        for test in range(mlp_n_tests):
            input_p = random.randint(1,16)
            input_f = random.randint(1,16)
            input = torch.randn(self.dataset_size, input_p, input_f).to(device)

            inputTens = torch.randn(16,input_p,input_f)
            MLP_arch = self.make_mlp(mlp_hidden_sizes, inputTens)
            
            score_testedArch, score_MLP, testedArch_penalty, MLP_penalty = self.get_scores(architecture, MLP_arch, input, get_penalties=True)
            print(f"Tested Arch penalty: {testedArch_penalty}, MLP penalty: {MLP_penalty}")
            scores.append((score_testedArch, score_MLP))
        
        n_wins = 0
        for score in scores:
            if score[0] > score[1]:
                n_wins += 1
        return scores, n_wins/mlp_n_tests

    def realDataSet_test(self, architecture, verbose=True, max_iter=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type=='cuda':
            max_batch_size = 2048
        else:
            max_batch_size = 32
        
        results = {}

        def batched_forward(executor, data, batch_size=2048):
            outputs = []
            with torch.no_grad():
                for i in range(0, len(data), batch_size):
                    batch = data[i:i+batch_size].to(device)
                    out = executor.forward(batch)
                    outputs.append(out[0].cpu())
            return [torch.cat(outputs, dim=0).to(device)]

        ################# 1 - California housing dataset #################

        # Load California housing dataset
        data = fetch_california_housing()
        X, y = data.data, data.target

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        # convert to 3D tensors
        train_input = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
        train_target = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1,1).to(device)

        test_input = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        test_target = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1,1).to(device)

        

        
        arch_copy = copy.deepcopy(architecture)
        arch_copy.reset_state()
        executor = Executor(arch_copy).to(device)

        # measure time taken for fitting
        start_fit_time = time.time()
        executor.fit(train_input, train_target, verbose=verbose, lr=0.001, max_iter=max_iter, batch_size=min(len(train_input),max_batch_size), patience = 10, min_delta = 1e-7, cpu = False)
        end_fit_time = time.time()
        fit_delay = end_fit_time - start_fit_time

        # measure time taken for testing
        start_test_time = time.time()
        test_output = batched_forward(executor, test_input)
        end_test_time = time.time()
        test_delay = end_test_time - start_test_time
        test_loss = torch.nn.functional.mse_loss(test_output[0], test_target)

        results['california_housing'] = {'test_loss': test_loss.item(), 'fit_delay': fit_delay, 'test_delay': test_delay}
        del executor, train_input, train_target, test_input, test_target
        torch.cuda.empty_cache()

        ################# 2 - MNIST #################

        # Load MNIST dataset

        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        
        # minmax scaling
        train_input_data = train_dataset.data.float().to(device) / 255.0
        test_input_data = test_dataset.data.float().to(device) / 255.0
        train_target_data = train_dataset.targets
        test_target_data = test_dataset.targets

        # convert to 3D tensors
        train_target_onehot = torch.nn.functional.one_hot(train_target_data, 10).float().unsqueeze(1).to(device)
        test_target_onehot = torch.nn.functional.one_hot(test_target_data, 10).float().unsqueeze(1).to(device)

        arch_copy = copy.deepcopy(architecture)
        arch_copy.reset_state()
        executor = Executor(arch_copy).to(device)
        # measure time taken for fitting
        start_fit_time = time.time()
        executor.fit(train_input_data, train_target_onehot, verbose=verbose, lr=0.001, max_iter=max_iter, batch_size=min(len(train_input_data),max_batch_size), patience = 10, min_delta = 1e-7, cpu = False)
        end_fit_time = time.time()
        fit_delay = end_fit_time - start_fit_time

        # measure time taken for testing
        start_test_time = time.time()
        test_output =batched_forward(executor, test_input_data)
        end_test_time = time.time()
        test_delay = end_test_time - start_test_time
        test_loss = torch.nn.functional.mse_loss(test_output[0], test_target_onehot).to(device)

        results['mnist'] = {'test_loss': test_loss.item(), 'fit_delay': fit_delay, 'test_delay': test_delay}
        del executor, train_input_data, train_target_onehot, test_input_data, test_target_onehot
        torch.cuda.empty_cache()

        ################# 3 - CIFAR10 #################

        # Load CIFAR10 dataset
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        
        train_input = torch.stack([img for img, _ in train_dataset])   # [50000, 3, 32, 32]
        train_input = train_input.permute(0, 2, 1, 3).contiguous().view(50000, 32, 96)

        train_target = torch.tensor([label for _, label in train_dataset])
        train_target_onehot = torch.nn.functional.one_hot(train_target, 10).float().unsqueeze(1).to(device)

        test_input = torch.stack([img for img, _ in test_dataset])     # [10000, 3, 32, 32]
        test_input = test_input.permute(0, 2, 1, 3).contiguous().view(10000, 32, 96)
        test_target = torch.tensor([label for _, label in test_dataset])
        test_target_onehot = torch.nn.functional.one_hot(test_target, 10).float().unsqueeze(1).to(device)

        arch_copy = copy.deepcopy(architecture)
        arch_copy.reset_state()
        executor = Executor(arch_copy).to(device)
        # measure time taken for fitting
        start_fit_time = time.time()
        executor.fit(train_input, train_target_onehot, verbose=verbose, lr=0.001, max_iter=max_iter, batch_size=min(len(train_input),max_batch_size), patience = 10, min_delta = 1e-7, cpu = False)
        end_fit_time = time.time()
        fit_delay = end_fit_time - start_fit_time

        # measure time taken for testing
        start_test_time = time.time()
        test_output = batched_forward(executor, test_input)
        end_test_time = time.time()
        test_delay = end_test_time - start_test_time
        test_loss = torch.nn.functional.mse_loss(test_output[0], test_target_onehot)

        results['cifar10'] = {'test_loss': test_loss.item(), 'fit_delay': fit_delay, 'test_delay': test_delay}
        del executor
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

        



                    
                    
                    


        
        

        