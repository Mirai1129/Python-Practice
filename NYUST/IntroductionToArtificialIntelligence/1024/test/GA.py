import random
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np


class GeneticAlgorithm:
    def __init__(
            self,
            pop_size: int,  # Population size
            generations: int,  # Number of generations for the algorithm
            mutation_rate: float,  # Gene mutation rate
            crossover_rate: float,  # Gene crossover rate
            tournament_size: int,  # Tournament size for selection
            elitism: bool,  # Whether to apply elitism strategy
            random_seed: Optional[int],  # Random seed for reproducibility
    ):
        # Students need to set the algorithm parameters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _init_population(self, M: int, N: int) -> List[List[int]]:
        """
        Initialize the population and generate random individuals, ensuring that every student is assigned at least one task.
        :param M: Number of students
        :param N: Number of tasks
        :return: Initialized population
        """
        population = []
        for _ in range(self.pop_size):
            individual = [-1] * N  # Create an individual with -1 indicating unassigned tasks
            tasks = list(range(N))

            # Assign at least one task to each student
            for student in range(M):
                task_idx = random.choice(tasks)
                individual[task_idx] = student
                tasks.remove(task_idx)

            # Randomly assign remaining tasks
            for task in tasks:
                individual[task] = random.randint(0, M - 1)

            population.append(individual)
        return population

    def _fitness(self, individual: List[int], student_times: np.ndarray) -> float:
        """
        Fitness function: calculate the fitness value of an individual ensuring each student gets at least one task.

        :param individual: Individual representing student assignments
        :param student_times: Time required for each student to complete each task
        :return: Fitness value (total time)
        """
        num_tasks = np.size(student_times, 1)
        task_assignments = [-1] * num_tasks
        assigned_students = set()
        total_time = 0

        # First pass: Assign tasks optimally
        for task in range(num_tasks):
            min_time = float('inf')
            best_student = -1

            # Find the student who can complete this task in minimum time
            for student in individual:
                time = student_times[student][task]
                if time < min_time:
                    min_time = time
                    best_student = student

            task_assignments[task] = best_student
            assigned_students.add(best_student)
            total_time += min_time

        # Second pass: Ensure all students have at least one task
        unassigned_students = set(individual) - assigned_students
        if unassigned_students:
            # For each unassigned student
            for student in unassigned_students:
                # Find the best task to reassign to this student
                min_additional_cost = float('inf')
                best_task = -1

                for task in range(num_tasks):
                    current_student = task_assignments[task]
                    current_time = student_times[current_student][task]
                    new_time = student_times[student][task]

                    # Calculate the additional cost of reassignment
                    additional_cost = new_time - current_time

                    # Check if this is the best reassignment so far
                    if additional_cost < min_additional_cost:
                        # Only consider this reassignment if the current student has multiple tasks
                        student_task_count = task_assignments.count(current_student)
                        if student_task_count > 1:
                            min_additional_cost = additional_cost
                            best_task = task

                if best_task != -1:
                    # Update total time and assignments
                    total_time += min_additional_cost
                    task_assignments[best_task] = student

        return total_time



    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """
        Use tournament selection to choose parents for crossover.
        :param population: Current population
        :param fitness_scores: Fitness scores for each individual
        :return: Selected parent
        """
        tournament = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
        best_individual = min(tournament, key=lambda x: x[1])[0]
        return best_individual

    def _crossover(self, parent1: List[int], parent2: List[int], M: int) -> Tuple[List[int], List[int]]:
        """
        Crossover: generate two offspring from two parents.
        :param parent1: Parent 1
        :param parent2: Parent 2
        :param M: Number of students
        :return: Generated offspring
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2  # No crossover; return parents as offspring

        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

        # Ensure each student is assigned at least one task
        for student in range(M):
            if student not in offspring1:
                task_to_reassign = random.choice([task for task in range(len(offspring1)) if offspring1[task] != student])
                offspring1[task_to_reassign] = student

        for student in range(M):
            if student not in offspring2:
                task_to_reassign = random.choice([task for task in range(len(offspring2)) if offspring2[task] != student])
                offspring2[task_to_reassign] = student

        return offspring1, offspring2

    def _mutate(self, individual: List[int], M: int) -> List[int]:
        """
        Mutation operation: randomly change some genes (task assignments) of the individual.
        :param individual: Individual
        :param M: Number of students
        :return: Mutated individual
        """
        for task_idx in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[task_idx] = random.randint(0, M - 1)

        # Ensure each student is assigned at least one task after mutation
        assigned_students = set(individual)
        for student in range(M):
            if student not in assigned_students:
                task_to_reassign = random.choice([task for task in range(len(individual)) if individual[task] != student])
                individual[task_to_reassign] = student

        return individual

    def __call__(self, M: int, N: int, student_times: np.ndarray) -> Tuple[List[int], int]:
        """
        Execute the genetic algorithm and return the optimal solution (allocation plan) and its total time cost.
        :param M: Number of students
        :param N: Number of tasks
        :param student_times: Time required for each student to complete each task
        :return: Optimal allocation plan and total time cost
        """
        # Initialize population
        population = self._init_population(M, N)
        fitness_values = []  # 用於記錄每次的適應度值

        for generation in range(self.generations):
            # Calculate fitness for the population
            fitness_scores = [self._fitness(individual, student_times) for individual in population]

            # 記錄每次的 fitness 值
            fitness_values.extend(fitness_scores)

            # Apply elitism (carry the best individual to the next generation)
            if self.elitism:
                best_individual = min(population, key=lambda ind: self._fitness(ind, student_times))
                next_population = [best_individual]
            else:
                next_population = []

            # Generate new population through selection, crossover, and mutation
            while len(next_population) < self.pop_size:
                # Selection
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)

                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2, M)

                # Mutation
                offspring1 = self._mutate(offspring1, M)
                offspring2 = self._mutate(offspring2, M)

                # Add offspring to the next population
                next_population.append(offspring1)
                if len(next_population) < self.pop_size:
                    next_population.append(offspring2)

            # Replace the old population with the new one
            population = next_population

        # 使用 Counter 找出出現最多次的適應度值
        most_common_fitness = Counter(fitness_values).most_common(1)
        if most_common_fitness:
            most_common_value = most_common_fitness[0][0]
            # 根據最常見的 fitness 值找出最佳個體
            best_individual = min(population, key=lambda ind: abs(self._fitness(ind, student_times) - most_common_value))
        else:
            best_individual = None  # 或者你可以設定為其他預設值

        total_time = self._fitness(best_individual, student_times) if best_individual else float('inf')

        return best_individual, int(total_time)


if __name__ == "__main__":
    def write_output_to_file(problem_num: int, total_time: int, filename: str = "results.txt") -> None:
        """
        Write results to a file and check if the format is correct.
        """
        print(f"Problem {problem_num}: Total time = {total_time}")

        if not isinstance(total_time, int):
            raise ValueError(f"Invalid format for problem {problem_num}. Total time should be an integer.")

        with open(filename, 'a') as file:
            file.write(f"Total time = {total_time}\n")


    # TODO: Define multiple test problems based on the examples and solve them using the genetic algorithm
    # Example problem 1 (define multiple problems based on the given example format)
    # M, N = 2, 3
    # student_times = [[3, 8, 6],
    #                  [5, 2, 7]]

    M1, N1 = 5, 5
    cost1 = [[8,8,4,1,2],
            [5,2,6,6,2],
            [7,10,2,1,5],
            [1,3,2,7,7],
            [8,11,9,8,9]]
    # total time = 15

    M2, N2 = 2, 3
    cost2 = [[9, 10, 11], # Tom
            [1, 2, 3]]    # Jerry
    # total time = 14

    M3, N3 = 8, 9
    cost3 = [[90, 100, 60, 5, 50, 1, 100, 80, 70],
             [100, 5, 90, 100, 50, 70, 60, 90, 100],
             [50, 1, 100, 70, 90, 60, 80, 100, 4],
             [60, 100, 1, 80, 70, 90, 100, 50, 100],
             [70, 90, 50, 100, 100, 4, 1, 60, 80],
             [100, 60, 100, 90, 80, 5, 70, 100, 50],
             [100, 4, 80, 100, 90, 70, 50, 1, 60],
             [1, 90, 100, 50, 60, 80, 100, 70, 5]]

    problems = [(M1, N1, np.array(cost1)),
                (M2, N2, np.array(cost2)),
                (M3, N3, np.array(cost3))]

    # Example for GA execution:
    # TODO: Please set the parameters for the genetic algorithm
    ga = GeneticAlgorithm(
        pop_size=100,  # 族群大小
        generations=2,  # 世代數
        mutation_rate=0.02,  # 突變機率
        crossover_rate=0.8,  # 交叉機率
        tournament_size=3,  # 錦標賽大小
        elitism=True,  # 使用菁英策略
        random_seed=42  # 隨機種子（可選）
    )

    # Solve each problem and immediately write the results to the file
    for i, (M, N, student_times) in enumerate(problems, 1):
        best_allocation, total_time = ga(M=M, N=N, student_times=student_times)
        write_output_to_file(i, total_time)

    print("Results have been written to results.txt")
