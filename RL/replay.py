import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Transition:
    s: torch.Tensor
    s_prime: torch.Tensor
    action: int
    reward : float # reward from taking action a in state s
    done : bool = False # whether the game is over


class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] # ring buffer for O(1) sampling and appending
        self.position = 0

    def push(self, transition: Transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        sample = random.sample(self.memory, min(batch_size, len(self.memory)))
        s_batch = torch.stack([transition.s for transition in sample]).to(device)
        s_prime_batch = torch.stack([transition.s_prime for transition in sample]).to(device)
        a_batch = torch.tensor([transition.action for transition in sample], dtype=torch.int64).to(device)
        r_batch = torch.Tensor([transition.reward for transition in sample]).to(device)
        done_batch = torch.Tensor([transition.done for transition in sample]).to(device)
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
    
    def display_memory_usage(self):
        total_memory = 0
        for transition in self.memory:
            total_memory += transition.s.element_size() * transition.s.numel()
            total_memory += transition.s_prime.element_size() * transition.s_prime.numel()

        total_memory_mb = total_memory / (1024 ** 2)  # Convert bytes to megabytes
        print(f"Memory usage of Experience Replay: {total_memory_mb:.2f} MB")
        return total_memory_mb

    def __len__(self):
        return len(self.memory)