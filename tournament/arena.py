import copy
import math

import torch
import random
import numpy as np
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

    def get_scores(self, arch_1, arch_2, input = None, get_penalties=False, outerfunction="sqrt", randomizeHP=False):
        device = torch.device('cuda' if torch.cuda.is_available() and not self.cpu else 'cpu')

        if device.type=='cuda':
            max_batch_size = 2048
        else:
            max_batch_size = 32

        # Generate random values for the hyperparameters
        if randomizeHP:
            random_batch_size = max_batch_size + (random.random()-0.5) * max_batch_size * 0.5 # more or less 25% of the batch size
            random_lr = random.choice([0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001,0.00005, 0.00001])
            random_patience = random.choice([6, 8, 10, 12, 15, 20, 25, 30])
            random_min_delta = random.choice([1e-5, 1e-6, 1e-7, 1e-8])
            random_max_iter = random.choice([100, 200, 300, 400, 500, 600, 700])
    
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
            train_size = int(self.dataset_size*self.train_test_split)

            input_p = random.randint(1,16)
            input_f = random.randint(1,16)
            input = torch.randn(self.dataset_size, input_p, input_f).to(device)

        train_size = int(self.dataset_size*self.train_test_split)
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
            score_1 = (1/(test_loss_1*( deltaK_1 ** self.pcp) ))**0.5
            score_2 = (1/(test_loss_2*( deltaK_2 ** self.pcp) ))**0.5

            del executors[0]
            del executors[0], learner_1, learner_2 #because executors[0] is our previous executors[1] (executors[0] was deleted, which made executors[1] become executors[0]))
            torch.cuda.empty_cache()
            if get_penalties:
                return score_1, score_2, deltaK_1, deltaK_2
            return score_1, score_2


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
        print(arch_scores)
        n_wins = 0
        for score in arch_scores:
            if score[0] == max(score):
                n_wins += 1
        return arch_scores, n_wins
            
    
    def make_mlp(self,hidden_sizes, inputTens):
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

        
        



                    
                    
                    


        
        

        