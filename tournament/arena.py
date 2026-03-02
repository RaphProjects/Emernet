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
        architectures = []
        executors = []
        # Generate the first architecture
        generator = Generator(generation_type=self.generation_type)
        firstarch = generator.generate(self.architecture_size)
        architectures.append(firstarch)
        for n_fight in range(self.n_fights):
            for gen_architecture in range(self.arena_contestants-1):
                # Generate a new architecture
                new_architecture = generator.generate(self.architecture_size)
                architectures.append(new_architecture)
                print(f"New architecture: {architectures[n_fight+gen_architecture+1]}")

            for arch in architectures:
                        executors.append(Executor(arch).to(device))

            # architectures is filled, now we evaluate them two by two (every possible pair)
            for i in range(self.arena_contestants-1): # TODO : create new executors for each pair
                for j in range(i+1, self.arena_contestants):
                    # Randomize the weights of the architectures
                    executors[i].randomize_weights()
                    executors[j].randomize_weights()

                    # create the input
                    train_size = int(self.dataset_size*self.train_test_split)

                    input_p = random.randint(1,16)
                    input_f = random.randint(1,16)
                    input = torch.randn(self.dataset_size, input_p, input_f).to(device)
                    train_input = input[:train_size]
                    test_input = input[train_size:]


                    # generate the outputs for each executor
                    output_i = executors[i].forward(input)
                    output_j = executors[j].forward(input)

                    train_target_i = output_j[0][:train_size].to(device)
                    train_target_j = output_i[0][:train_size].to(device)

                    test_target_i = output_j[0][train_size:].to(device)
                    test_target_j = output_i[0][train_size:].to(device)

                    # make new executors and fit them
                    learner_i = Executor(architectures[i]).to(device)
                    learner_j = Executor(architectures[j]).to(device)
                    learner_i.fit(train_input, train_target_i.detach(), verbose=self.verbose, lr=0.01, max_iter=100, batch_size=16, patience = 8, min_delta = 1e-7, cpu = False)
                    learner_j.fit(train_input, train_target_j.detach(), verbose=self.verbose, lr=0.01, max_iter=100, batch_size=16, patience = 8, min_delta = 1e-7, cpu = False)

                    with torch.no_grad():
                        pred_i = learner_i.forward(test_input)[0]
                        pred_j = learner_j.forward(test_input)[0]
                        
                        test_loss_i = torch.nn.functional.mse_loss(pred_i, test_target_i).item()
                        test_loss_j = torch.nn.functional.mse_loss(pred_j, test_target_j).item()

                        if self.verbose:
                            print(f"Architecture I{n_fight} :")
                            architectures[i].describe()
                            print(f"Architecture J{n_fight} :")
                            architectures[j].describe()
                            print(f"Test loss for executor {i}: {test_loss_i}")
                            print(f"Test loss for executor {j}: {test_loss_j}")

                    
                    
                    


        
        

        