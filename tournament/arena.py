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
    def __init__(self, n_fights=1, architecture_size=16, arena_contestants=3, dataset_size = 256+64, train_test_split= 0.7, generation_type="agnostic", verbose=True, report=False):
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
    


    def start(self):
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
        
        for n_fight in range(self.n_fights):
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
                    print(f"Generating executors for architectures {i} and {j}")
                    executors = []
                    # create the executors  
                    executors.append(Executor(copy.deepcopy(architectures[i])).to(device))
                    executors.append(Executor(copy.deepcopy(architectures[j])).to(device))
                    # Randomize the weights of the architectures
                    executors[0].randomize_weights()
                    executors[1].randomize_weights()

                    # create the input
                    train_size = int(self.dataset_size*self.train_test_split)

                    input_p = random.randint(1,16)
                    input_f = random.randint(1,16)
                    input = torch.randn(self.dataset_size, input_p, input_f).to(device)
                    train_input = input[:train_size]
                    test_input = input[train_size:]


                    # generate the outputs for each executor
                    output_i = executors[0].forward(input)
                    output_j = executors[1].forward(input)

                    train_target_i = output_j[0][:train_size].to(device)
                    train_target_j = output_i[0][:train_size].to(device)

                    test_target_i = output_j[0][train_size:].to(device)
                    test_target_j = output_i[0][train_size:].to(device)

                    # make new executors and fit them
                    learner_i = Executor(copy.deepcopy(architectures[i])).to(device)
                    learner_j = Executor(copy.deepcopy(architectures[j])).to(device)
                    learner_i.fit(train_input, train_target_i.detach(), verbose=self.verbose, lr=0.01, max_iter=100, batch_size=min(train_size,2048), patience = 8, min_delta = 1e-7, cpu = False)
                    learner_j.fit(train_input, train_target_j.detach(), verbose=self.verbose, lr=0.01, max_iter=100, batch_size=min(train_size,2048), patience = 8, min_delta = 1e-7, cpu = False)

                    with torch.no_grad():
                        pred_i = learner_i.forward(test_input)[0]
                        pred_j = learner_j.forward(test_input)[0]
                        
                        test_loss_i = torch.nn.functional.mse_loss(pred_i, test_target_i).item()
                        test_loss_j = torch.nn.functional.mse_loss(pred_j, test_target_j).item()
                        
                        # compute the scores
                        K_i = math.log2(max(2,architectures[i].parameter_count()))
                        K_j = math.log2(max(2,architectures[j].parameter_count()))
                        deltaK_i = (K_i/K_j) + math.exp(-K_i*K_j)
                        deltaK_j = (K_j/K_i) + math.exp(-K_j*K_i)
                        score_i = test_loss_i*deltaK_i
                        score_j = test_loss_j*deltaK_j

                        if score_i > score_j:
                            scores[i] = scores[i] + 1
                        else:
                            scores[j] = scores[j] + 1

                        del executors[0]
                        del executors[0], learner_i, learner_j #because executors[0] is our previous executors[1]
                        torch.cuda.empty_cache()
                        if self.verbose:
                            print(f"Architecture {i} of round {n_fight} :")
                            architectures[i].describe()
                            print(f"Architecture {j} of round {n_fight} :")
                            architectures[j].describe()
                            print(f"Test loss for executor {i}: {test_loss_i}")
                            print(f"Test loss for executor {j}: {test_loss_j}")
            # Fight loop
            max_score_id = scores.index(max(scores))
            if max_score_id == 0: # because the first architecture is always the previous best
                winner_scores[current_winner_id] += scores[max_score_id]
            else:
                winners.append(architectures[max_score_id])
                winner_scores.append(scores[max_score_id])
                current_winner_id += 1
            current_best = copy.deepcopy(architectures[max_score_id])
        if self.verbose:
            print(f"Winners Scores: {winners}")
            print(f"Final scores: {scores}")
            print(f"Final architecture: ")
            current_best.describe()
        return winner_scores, winners



                    
                    
                    


        
        

        