def check_dtype_support(dtype):
    try:
        torch.tensor(1, dtype=dtype, device=device)
        # print('sukses bro')
        return True
    except TypeError:
        # print('ada error bro')
        return False

# Check support for float64
supports_float64 = check_dtype_support(torch.float64)
print(f"Float64 support: {supports_float64}")

# Check support for float32
supports_float32 = check_dtype_support(torch.float32)
print(f"Float32 support: {supports_float32}")

def simple_fx(x):
    if(check_dtype_support(torch.float64)):
        # x = torch.tensor(x, dtype=torch.float64, device=device)
        x.to(torch.float64)
        # x_in = torch.tensor(x,device = device)
        pass
    else:
        # x = x.astype('float32')
        # x = torch.tensor(x, dtype=torch.float32, device=device)
        x.to(torch.float32)
        # x_in = torch.tensor(x,device = device)
    # print(x)
    # hasil = -np.power(x,2) + 14*x - 13
    # print(hasil)
    # return -hasil
    # return -(np.sum(-np.power(x,2) + 14*np.array(x) - 13))
    
    #     if x.dim() == 1:
    #       x = x.unsqueeze(0)
    #     dim = x.size(dim=-1)

    #     hasil = (-torch.pow(x[:,0],2) + 14*x[:,0] - 13).reshape(-1, 1)
    #     return -hasil
    
    #if device in ('cuda', 'cpu'):
    #    # bounds = torch.tensor(bounds, device = set_device)
    #    # min_bounds = torch.tensor(min_bounds, device = set_device)
    #    # max_bounds = torch.tensor(max_bounds, device = set_device)
    #    # x_in = torch.tensor(x,device = device)
    #    pass
    #elif device == 'mps':
    #    # bounds = np.asarray(bounds, dtype=np.float32)
    #    # min_bounds = np.asarray(min_bounds, dtype=np.float32)
    #   # max_bounds = np.asarray(max_bounds, dtype=np.float32)

    #    # atau dengan

    #    # x = x.astype('float32')

     #   x_in = torch.tensor(x, device = device)
    
    # return -(torch.sum(-torch.pow(x_in[:,0],2) + 14*x_in[:,0] - 13))
    # return -(-torch.pow(x_in[:,0],2) + 14*x_in[:,0] - 13)
    # return -(-torch.pow(x[:,0],2) + 14*x[:,0] - 13).reshape(-1, 1)
    return -(-torch.pow(x[:,0],2) + 14*x[:,0] - 13)

# Function to perform experiments and save results
# def run_experiments(test_functions, optimization_algorithms, bounds):
def run_experiments(test_functions, optimization_algorithms, bounds_test_functions):
    results = []
    # for func in test_functions:
    for func,bound in zip(test_functions,bounds_test_functions):
        func_results = {'function': func.__name__, 'results': []}
        for optimizer in optimization_algorithms:
            print(func.__name__)
            print(bound)
            print(optimizer.__name__)
            print()
            
            start_time = time.time()
            # best_solution, best_fitness, all_solutions = optimizer(func, bounds)
            best_solution_gpu, best_fitness_gpu, all_solutions_gpu = optimizer(func, bound)
            
            best_solution, best_fitness, all_solutions = \
            best_solution_gpu.cpu().numpy().flatten(),\
            best_fitness_gpu.cpu().numpy().flatten(),\
            all_solutions_gpu.cpu().numpy().flatten()
            
            
            
            #best_position_before_np = np.round(np.array(gBest_init.cpu()),4)
            #best_position =  mydenorm_torch_v2(gBest_init, min_bounds, max_bounds, bounds, set_device)
            #best_position_after_np = best_position.cpu().numpy().flatten()
            
            running_time = time.time() - start_time
            # fitness_values = [-func(sol) for sol in all_solutions]
            # fitness_values_gpu = [-func(sol) for sol in all_solutions_gpu]
            # fitness_values_gpu = [-func(all_solutions_gpu)]
            fitness_values_gpu = -func(all_solutions_gpu)
            fitness_values = fitness_values_gpu.cpu().numpy().flatten()

            # func_results['results'].append({
            #     'algorithm_name': optimizer.__name__,
            #     'best_solution': best_solution,
            #     'best_fitness': best_fitness,
            #     'median_fitness': np.median(fitness_values),
            #     'worst_fitness': np.max(fitness_values),
            #     'mean_fitness': np.mean(fitness_values),
            #     'std_fitness': np.std(fitness_values),
            #     'rank': None,
            #     'p_value': None,
            #     'significant': None,
            #     'running_time': running_time
            # })
            
            func_results['results'].append({
                'algorithm_name': optimizer.__name__,
                'best_solution': best_solution,
                'best_fitness': -best_fitness,
                'median_fitness': np.median(fitness_values),
                'worst_fitness': np.max(fitness_values),
                'mean_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'running_time': running_time
            })

        results.append(func_results)

    return results

# bounds_test_functions

# Function to save results to CSV and Excel files
def save_results_to_file(results, file_format, path_to_save=None):
    for func_result in results:
        df = pd.DataFrame(func_result['results'])
        if path_to_save == None:
            filename = f"{func_result['function']}_{file_format}_results"
        else:
            filename = path_to_save+f"/{func_result['function']}_{file_format}_results"
        df.to_csv(f"{filename}.csv", index=False)
        df.to_excel(f"{filename}.xlsx", index=False)

# Function to create a chart and save it to a PDF file
def save_chart_to_pdf(results,path_to_save=None):
    if path_to_save == None:
        pdf_pages = PdfPages("chart_results.pdf")
    else:
        pdf_pages = PdfPages(path_to_save+"/chart_results.pdf")
        
        
    # patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    patterns = [ "\\" , "." , "|" , "-" , "+" , "x", "o", "O", "/", "*" ]

    # ax1 = fig.add_subplot(111)
    # for i in range(len(patterns)):
    #     ax1.bar(i, 3, color='green', edgecolor='black', hatch=patterns[i])

    nama_all_alg = [results[0]['results'][i]['algorithm_name'] for i in range(len(results[0]['results']))]
    len_nama_all_alg = len(nama_all_alg)
    # print(nama_all_alg)

    for func_result in results:
        plt.figure(figsize=(10, 6))
        
        # plt.bar(func_result['results'][0]['algorithm_name'], \
        #         func_result['results'][0]['best_fitness'], \
        #         label='Best Fitness', edgecolor='white', \
        #         hatch=patterns[0])
        # plt.bar(func_result['results'][1]['algorithm_name'], \
        #         func_result['results'][1]['best_fitness'], \
        #         label='Best Fitness', edgecolor='white',\
        #         hatch=patterns[1])
        # plt.bar(func_result['results'][2]['algorithm_name'], \
        #         func_result['results'][2]['best_fitness'], \
        #         label='Best Fitness', edgecolor='white',\
        #         hatch=patterns[3])
        
        for j in range(len_nama_all_alg):
            # print(j)
            plt.bar(func_result['results'][j]['algorithm_name'], \
                    func_result['results'][j]['best_fitness'], \
                    label='Best Fitness', edgecolor='white', \
                    hatch=patterns[-j])
        
        
        plt.title(f"Best Fitness for {func_result['function']}")
        plt.xlabel("Algorithm")
        plt.ylabel("Fitness Value")
        plt.legend()
        pdf_pages.savefig()
        plt.close()

    pdf_pages.close()

# Run experiments
# test_functions = [rastrigin, ackley, lambda x: weierstrass(x, 0.5, 3, 20), cosine]
# test_functions = [rastrigin, ackley, cosine, simple_fx, problem_obj_1]
# test_functions = [cosine, simple_fx, problem_obj_1, problem_obj_1_cara_lain]
test_functions = [simple_fx, problem_obj_1, problem_obj_1_cara_lain, problem_1, problem_2]
optimization_algorithms = [pso, ptvpso, bgra]
bounds = [(-5.12, 5.12), (-5, 5), (-0.5, 0.5), (-5, 5)]  # 4 Dim
bounds_cos = [(-2*3.14, 2*3.14)]  # Adjust bounds based on the functions => 1 Dim


# bounds_simple_fx = np.array([[0], [15]])
bounds_simple_fx = [(0, 15)]
bound_problem_obj_1 = [(0,50),(0,50)]

bounds_problem_2 = [(-10, 10), (-10, 10)]

# bounds_test_functions = [bounds_rastrigin=bounds, bounds_ackley=bounds, bounds_cosine=bounds_cos]
# bounds_test_functions = [bounds, bounds, bounds_cos, bounds_simple_fx, bound_problem_obj_1]
# bounds_test_functions = [bounds_cos, bounds_simple_fx, bound_problem_obj_1, bound_problem_obj_1]
bounds_test_functions = [bounds_simple_fx, bound_problem_obj_1, \
                         bound_problem_obj_1, bounds_simple_fx, \
                         bounds_problem_2]

# results = run_experiments(test_functions, optimization_algorithms, bounds)
results = run_experiments(test_functions, optimization_algorithms, bounds_test_functions)

# Display results in a table
print("\nResults:")
print("--------------------------------------------------------------------------------------------------------------------------------------------------------")
print("| Function           | Algorithm      | Best Solution                   | Best Fitness | Median Fitness | Worst Fitness | Mean Fitness | Std Fitness | Running Time (s) |")
print("--------------------------------------------------------------------------------------------------------------------------------------------------------")
for func_result in results:
    for result in func_result['results']:
        print(f"| {func_result['function']:<20} | {result['algorithm_name']:<15} | {result['best_solution']} | {result['best_fitness']:<12} | {result['median_fitness']:<15} | {result['worst_fitness']:<13} | {result['mean_fitness']:<12} | {result['std_fitness']:<11} | {result['running_time']:<17.5f} |")
print("--------------------------------------------------------------------------------------------------------------------------------------------------------")

# for func_result in results:
#     for result in func_result['results']:
#         print(f"| {func_result['function']:<20} | {result['algorithm']:<11} | {result['best_solution']} | {result['best_fitness']:<12} | {result['median_fitness']:<15} | {result['worst_fitness']:<13} | {result['mean_fitness']:<12} | {result['std_fitness']:<11} | {result['running_time']:<17.5f} |")
# print("--------------------------------------------------------------------------------------------------------------------------------------------------------")

# # Statistical analysis (p-values)
# for i in range(len(test_functions)):
#     for j in range(len(optimization_algorithms)):
#         for k in range(j + 1, len(optimization_algorithms)):
#             algorithm1 = results[i]['results'][j]['algorithm_name']
#             algorithm2 = results[i]['results'][k]['algorithm_name']
#             fitness_values1 = [results[i]['results'][j]['best_fitness']]  # You can use other fitness values for comparison
#             fitness_values2 = [results[i]['results'][k]['best_fitness']]
#             t_stat, p_value = ttest_ind(fitness_values1, fitness_values2)
#             print(f"\nStatistical Analysis for {results[i]['function']} using {algorithm1} and {algorithm2}:")
#             print(f"P-value: {p_value}")

# Save results to CSV and Excel files
save_results_to_file(results, 'final')

# Save chart to PDF
save_chart_to_pdf(results)
