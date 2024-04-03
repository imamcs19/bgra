# bacterial gene recombination algorithm (BGRA)
# source code by: Imam Cholissodin | imamcs@ub.ac.id | Filkom UB
# 
# ref. article / paper: https://www.sciencedirect.com/science/article/abs/pii/S0096300314000174
# title: A bacterial gene recombination algorithm for solving constrained optimization problems | Tsung-Jung Hsieh
#
# 
# def bgra(objective_func, bounds, num_iterations=num_iterations_all, population_size=pop_size_all, swim_length=4, tumble_rate=0.1, elimination_rate=0.25, dispersal_rate=0.1, chemotaxis_length=100):
def bgra(objective_func, bounds, num_iterations=num_iterations_all, \
         population_size=pop_size_all, \
         select_ratio=0.6, r=0.1, \
         MGN=100):
    
    # select_ratio = [0.4,0.6]
    # r ={0.001, 0.005, 0.05, 0.01 and 0.1}
    M = population_size
    dim = len(bounds)
    d = dim
    
    batas_count = MGN * r # sebagai batas tidak ada perbaikan dari individu

    # init populasi
    population = torch.rand((population_size, dim), device=device) * \
    (torch.tensor(bounds, device = device)[:, 1] - \
     torch.tensor(bounds, device = device)[:, 0]) + \
    torch.tensor(bounds, device = device)[:, 0]
    
    # fitness_values = [objective_func(individual) for individual in population]
    # fitness_values = objective_func(population)
    fitness_values = torch.zeros((population_size, ), device=device)
    count = torch.zeros_like(fitness_values)

    for iteration in range(num_iterations):
        
        # Transformation / transformasi ----------------------
        # ----------------------------- ----------------------
    
        for i in range(population_size):
            
            flag_rand_k=True
            while(flag_rand_k):
                # memilih 1 individu ke-k secara random, untuk x
                # dimana population[k] != population[i]
                k = torch.randint(population_size, (1,)).item()
                if(k==i):
                    flag_rand_k = True
                else:
                    flag_rand_k = False

            # if(i!=k):
            # print()
            new_bacterium_Zis = torch.zeros_like(population[i])
            for s in range(dim):
                # rand_pengali low=-1, high=1
                rand_pengali = (torch.rand((1, ), device=device) * \
                             (1 - \
                              (-1)) + \
                             (-1)).item()
                new_bacterium_Zis[s] = population[i][s] + \
                rand_pengali * \
                (population[k][s] - population[i][s])


            new_bacterium_Zis = torch.clamp(new_bacterium_Zis, \
                                            torch.tensor(bounds, device = device)[:, 0], \
                                            torch.tensor(bounds, device = device)[:, 1])


            # lakukan local search
            fitness_new = objective_func(new_bacterium_Zis.reshape(-1,1))
            fitness_old = objective_func(population[i].reshape(-1,1))
            if(fitness_new >= fitness_old):
                # replace yang lama dgn yang baru
                population[i] = new_bacterium_Zis
                count[i] = 0
            else:
                count[i] +=1
        
        # memilih 1 individu ke-k secara random, untuk x
        # dimana population[k] != population[i]
        #k = torch.randint(population_size, (1,)).item()
        #for i in range(population_size):

         #   if(i!=k):
         #       # print()
         #       new_bacterium_Zis = torch.zeros_like(population[i])
         #       for s in range(dim):
         #           # rand_pengali low=-1, high=1
         #           rand_pengali = (torch.rand((1, ), device=device) * \
         #                        (1 - \
         #                         (-1)) + \
         #                        (-1)).item()
         #           new_bacterium_Zis[s] = population[i][s] + \
         #           rand_pengali * \
         #           (population[k][s] - population[i][s])


         #       new_bacterium_Zis = torch.clamp(new_bacterium_Zis, \
         #                                       torch.tensor(bounds, device = device)[:, 0], \
         #                                       torch.tensor(bounds, device = device)[:, 1])

                
                # lakukan local search
         #       fitness_new = objective_func(new_bacterium_Zis.reshape(-1,1))
         #       fitness_old = objective_func(population[i].reshape(-1,1))
         #     if(fitness_new >= fitness_old):
                   # replace yang lama dgn yang baru
         #          population[i] = new_bacterium_Zis
         #      else:
         #          count[i] +=1
                    
                    
            
        # end Transformation / transformasi ----------------------
        
        # ----------------------
        # ----------------------
        # ----------------------
        
        # Transduction / transduksi ----------------------
        # ----------------------------- ----------------------

        for i in range(population_size):
            infected_bacterium_Xij = torch.zeros_like(population[i])
            for s in range(dim):
                # rand_pengali low=-1, high=1
                rand_pengali = (torch.rand((1, ), device=device) * \
                             (1 - \
                              (-1)) + \
                             (-1)).item()
                infected_bacterium_Xij[s] = population[i][s] + \
                rand_pengali * population[i][s]

            infected_bacterium_Xij = torch.clamp(infected_bacterium_Xij, \
                                            torch.tensor(bounds, device = device)[:, 0], \
                                            torch.tensor(bounds, device = device)[:, 1])


            # lakukan local search
            fitness_new = objective_func(infected_bacterium_Xij.reshape(-1,1))
            fitness_old = objective_func(population[i].reshape(-1,1))
            if(fitness_new >= fitness_old):
                # replace yang lama dgn yang baru
                population[i] = infected_bacterium_Xij
                count[i] = 0
            else:
                count[i] +=1
        # end Transduction / transduksi ----------------------
        
        # ----------------------
        # ----------------------
        # ----------------------
        
        # Conjugation / konjugasi ----------------------
        # ----------------------------- ----------------------
        fitness_values = objective_func(population)
        for i in range(population_size):
            #hitung nilai prob. tiap individu
            Pi = fitness_values/torch.sum(fitness_values)
            
            # urutkan hasil Pi, secara ascending, krn fitness 
            # semakin kecil nilai objective_func semakin baik
            # 
            # Pi_sortAsc, sorted_indices = torch.sort(Pi, descending=False)
            sorted_indices = torch.argsort(Pi)
                
        level_int, level_float = int(M * select_ratio), M * select_ratio
        
        for i in range(level_int): # superior group
            p_i = sorted_indices[i]

            flag_rand_p_g=True
            while(flag_rand_p_g):
                # memilih 1 individu ke-p_g secara random, untuk x
                # dimana population[p_g] != population[p_i]
                p_g = torch.randint(population_size, (1,)).item()
                if(p_g==p_i):
                    flag_rand_p_g = True
                else:
                    flag_rand_p_g = False
            
            # print()
            new_bacterium_Vp_i = torch.zeros_like(population[p_i])
            
            # memilih 1 batasan dimensi ke-j secara random, untuk V
            # untuk dilakukan pembaruan dari indeks dimensi
            # 0, 1, .., (j-1), j
            j = torch.randint(dim, (1,)).item()
            for s in range(dim):
                if(s<=j): # dilakukan update
                    # rand_pengali low=-1, high=1
                    rand_pengali = (torch.rand((1, ), device=device) * \
                                 (1 - \
                                  (-1)) + \
                                 (-1)).item()
                    new_bacterium_Vp_i[s] = population[p_i][s] + \
                    rand_pengali * \
                    (population[p_g][s] - population[p_i][s])
                
                else: # disamakan dgn nilai sebelumnya
                    new_bacterium_Vp_i[s] = population[p_i][s]


            new_bacterium_Vp_i = torch.clamp(new_bacterium_Vp_i, \
                                            torch.tensor(bounds, device = device)[:, 0], \
                                            torch.tensor(bounds, device = device)[:, 1])


            # lakukan local search
            fitness_new = objective_func(new_bacterium_Vp_i.reshape(-1,1))
            fitness_old = objective_func(population[p_i].reshape(-1,1))
            if(fitness_new >= fitness_old):
                # replace yang lama dgn yang baru
                population[p_i] = new_bacterium_Vp_i
                count[p_i] = 0
            else:
                count[p_i] +=1
                
        t=0
        for i in range(level_int,population_size): # inferior group
            # print()
            
            p_i = sorted_indices[i]
            p_t = sorted_indices[t]
            population[p_i] = population[p_t]
            
            t=torch.remainder(t+1,level_int).item()
            
            # if(level_float < 0.5 && t > level_int):
            #     t = 0
            
                
        # end Conjugation / konjugasi ----------------------
        
        # ----------------------
        # ----------------------
        # ----------------------
        
        # Artificial - made ----------------------
        # ----------------------------- ----------------------

        for i in range(population_size):
            if(count[i] > batas_count):
                artificial_infected_bacterium_Xij = torch.zeros_like(population[i])
                for s in range(dim):
                    # b_rand_pengali low=-0.5, high=0.5
                    b_rand_pengali = (torch.rand((1, ), device=device) * \
                                 (0.5 - \
                                  (-0.5)) + \
                                 (-0.5)).item()
                    artificial_infected_bacterium_Xij[s] = population[i][s] + \
                    b_rand_pengali * population[i][s]

                artificial_infected_bacterium_Xij = torch.clamp(artificial_infected_bacterium_Xij, \
                                                torch.tensor(bounds, device = device)[:, 0], \
                                                torch.tensor(bounds, device = device)[:, 1])
                population[i] = artificial_infected_bacterium_Xij
                count[i] = 0

            
                
        # end Conjugation / konjugasi ----------------------
        
        # ----------------------
        # ----------------------
        # ----------------------
        
    fitness_values = objective_func(population)
    
    # Return the best individual and its fitness value
    best_index = torch.argmin(fitness_values)
    best_solution = population[best_index]
    best_fitness = fitness_values[best_index]

    return best_solution, best_fitness, population
