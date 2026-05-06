import random
import csv
import torch
from supernet import GNNSuperNet
from hw_predictor import HWPredictor

def multi_objective_score(accuracy, latency, memory, lat_constraint, mem_constraint):
    if latency > lat_constraint or memory > mem_constraint:
        return 0.0
    alpha, beta = 1.0, 0.5 
    efficiency_penalty = (latency / lat_constraint) + (memory / mem_constraint)
    return (alpha * accuracy) - (beta * efficiency_penalty)

def generate_random_architecture(num_positions):
    return [[random.randint(0,1), random.randint(0,2), random.randint(0,1)] for _ in range(num_positions)]

def evolutionary_search(supernet, predictor, constraints, max_iterations=200, pop_size=20):
    print("[INFO] Starting Multi-Stage Search...")
    lat_c, mem_c = constraints
    
    # Open CSV for logging
    csv_file = open("search_results.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Iteration", "Accuracy", "Latency_ms", "Memory_MB", "Score"])
    
    population = [generate_random_architecture(12) for _ in range(pop_size)]
    
    for iteration in range(max_iterations):
        scored_population = []
        for arch in population:
            lat = predictor.predict_latency(arch)
            mem = predictor.predict_peak_memory(arch)
            acc = random.uniform(85.0, 93.0) # Simulated accuracy
            
            score = multi_objective_score(acc, lat, mem, lat_c, mem_c)
            scored_population.append((score, arch, acc, lat, mem))
            
            # Log valid architectures to CSV
            if score > 0:
                csv_writer.writerow([iteration, acc, lat, mem, score])
            
        scored_population.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [x[1] for x in scored_population[:pop_size//2]]
        
        new_population = top_candidates.copy()
        while len(new_population) < pop_size:
            parent = random.choice(top_candidates)
            mutated = parent.copy()
            mut_pos = random.randint(0, 11)
            mutated[mut_pos] = [random.randint(0,1), random.randint(0,2), random.randint(0,1)]
            new_population.append(mutated)
            
        population = new_population
        if iteration % 50 == 0:
            print(f"Iteration {iteration} | Best Score: {scored_population[0][0]:.4f}")
            
    csv_file.close()
    return scored_population[0][1]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    supernet = GNNSuperNet(num_positions=12).to(device)
    predictor = HWPredictor(target_device="jetson")
    
    # Constraints: 30ms latency, 100MB peak memory
    constraints = (30.0, 100.0) 
    
    best_arch = evolutionary_search(supernet, predictor, constraints, max_iterations=200, pop_size=20)
    print("\n[SUCCESS] Search finished! Data logged to 'search_results.csv'.")
    print("Optimal Architecture Encoding:", best_arch)
