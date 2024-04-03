def ptvpso(objective_func, bounds, num_particles=pop_size_all, num_iterations=num_iterations_all, inertia_weight=0.7, cognitive_weight=1.5, social_weight=1.5):
    # Set default device to CUDA if available, else use CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set default
    # ---------------------
    # inertia_weight=0.5
    # cognitive_param=1.5
    # social_param=2.0
    # ---------------------
    
    # set param utk ptvpso, (Ratnaweera, 2004), (Chen at all, 2011)
    # ---------------------
    # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # skala_opt = 10
    # options = {'c1i': 2.5/skala_opt, \
    #          'c1f': 0.5/skala_opt, \
    #          'c2i':0.5/skala_opt, \
    #          'c2f':2.5/skala_opt, \
    #          'wmin': 0.4/skala_opt, \
    #          'wmax': 0.9/skala_opt}
    
    skala_opt = 10
    wmin=0.4/skala_opt
    wmax=0.9/skala_opt
    c1i=2.5/skala_opt
    c1f=0.5/skala_opt
    c2i=0.5/skala_opt
    c2f=2.5/skala_opt
    # ---------------------

    dim = len(bounds)
    
    # Initialize particle positions and velocities within the specified bounds
    particles_position = torch.rand((num_particles, dim), device=device) * \
    (torch.tensor(bounds, device=device)[:, 1] - \
     torch.tensor(bounds, device=device)[:, 0]) + \
    torch.tensor(bounds, device=device)[:, 0]
    particles_velocity = torch.rand((num_particles, dim), device=device)  # Initial random velocities
    
    # Initialize personal best positions and corresponding fitness values
    personal_best_position = particles_position.clone()
    personal_best_fitness = objective_func(personal_best_position)
    
    # Initialize global best position and corresponding fitness value
    global_best_index = torch.argmin(personal_best_fitness)
    global_best_position = personal_best_position[global_best_index].clone()
    global_best_fitness = personal_best_fitness[global_best_index]
    
    max_iter = num_iterations
    curr_iter = 0

    for iteration in range(num_iterations):
        # Update particle velocities and positions
        r1, r2 = torch.rand((num_particles, dim), device=device), torch.rand((num_particles, dim), device=device)
        
        # TVIW dan TVAC base paper PTVPSO
        w = wmin + ((wmax-wmin)*((max_iter-curr_iter)/max_iter))
        c1 = ((c1f-c1i)*(curr_iter/max_iter)) + c1i
        c2 = ((c2f-c2i)*(curr_iter/max_iter)) + c2i
        
        curr_iter = curr_iter + 1
        
        # velocity = (inertia_weight * velocity +
        #             cognitive_param * r1 * (personal_best - swarm) +
        #             social_param * r2 * (global_best - swarm))
        
        w_inertia_weight = w
        c1_cognitive_param = c1
        c2_social_param = c2
        
        #cognitive_component = \
        #cognitive_weight * r1 * (personal_best_position - particles_position)
        
        #social_component = \
        #social_weight * r2 * (global_best_position - particles_position)
        
        # particles_velocity = inertia_weight * particles_velocity + cognitive_component + social_component
        
        cognitive_component = \
        c1_cognitive_param * r1 * (personal_best_position - particles_position)
        
        social_component = \
        c2_social_param * r2 * (global_best_position - particles_position)
        
        particles_velocity = w_inertia_weight * particles_velocity + cognitive_component + social_component
        particles_position += particles_velocity
        
        # Ensure particles stay within the specified bounds
        particles_position = torch.clamp(particles_position, \
                                         torch.tensor(bounds, device=device)[:, 0], \
                                         torch.tensor(bounds, device=device)[:, 1])
        
        # Update personal best positions and fitness values
        current_fitness = objective_func(particles_position)
        update_indices = current_fitness < personal_best_fitness
        
        personal_best_position[update_indices] = particles_position[update_indices]
        personal_best_fitness[update_indices] = current_fitness[update_indices]
        
        # Update global best position and fitness value
        global_best_index = torch.argmin(personal_best_fitness)
        global_best_position = personal_best_position[global_best_index].clone()
        global_best_fitness = personal_best_fitness[global_best_index]
    
    return global_best_position, global_best_fitness, particles_position

# def ptvpso(objective_func, bounds, num_particles=pop_size_all, num_iterations=num_iterations_all):
    
#     # set default
#     # ---------------------
#     # inertia_weight=0.5
#     # cognitive_param=1.5
#     # social_param=2.0
#     # ---------------------
    
#     # set param utk ptvpso, (Ratnaweera, 2004), (Chen at all, 2011)
#     # ---------------------
#     # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
#     # skala_opt = 10
#     # options = {'c1i': 2.5/skala_opt, \
#     #          'c1f': 0.5/skala_opt, \
#     #          'c2i':0.5/skala_opt, \
#     #          'c2f':2.5/skala_opt, \
#     #          'wmin': 0.4/skala_opt, \
#     #          'wmax': 0.9/skala_opt}
    
#     skala_opt = 10
#     wmin=0.4/skala_opt
#     wmax=0.9/skala_opt
#     c1i=2.5/skala_opt
#     c1f=0.5/skala_opt
#     c2i=0.5/skala_opt
#     c2f=2.5/skala_opt
#     # ---------------------
    
# #     print('bounds \n', bounds)
# #     print(type(bounds))
    
#     dim = len(bounds)
#     # swarm = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(num_particles, dim))
#     swarm = np.random.uniform(low=np.array(bounds)[:, 0], high=np.array(bounds)[:, 1], size=(num_particles, dim))
    
   
    
#     velocity = np.random.uniform(low=-1, high=1, size=(num_particles, dim))
#     personal_best = swarm.copy()
#     global_best_idx = np.argmin([objective_func(p) for p in swarm])
#     global_best = swarm[global_best_idx].copy()

#     # for _ in range(num_iterations):
#     max_iter = num_iterations
#     curr_iter = 0
#     for _ in range(num_iterations):
#         # Update velocity and position
#         r1, r2 = np.random.rand(num_particles, dim), np.random.rand(num_particles, dim)

#         # TVIW dan TVAC base paper PTVPSO
#         w = wmin + ((wmax-wmin)*((max_iter-curr_iter)/max_iter))
#         c1 = ((c1f-c1i)*(curr_iter/max_iter)) + c1i
#         c2 = ((c2f-c2i)*(curr_iter/max_iter)) + c2i
        
#         curr_iter = curr_iter + 1
        
#         # velocity = (inertia_weight * velocity +
#         #             cognitive_param * r1 * (personal_best - swarm) +
#         #             social_param * r2 * (global_best - swarm))
        
#         w_inertia_weight = w
#         c1_cognitive_param = c1
#         c2_social_param = c2
        
#         velocity = (w_inertia_weight * velocity +
#                     c1_cognitive_param * r1 * (personal_best - swarm) +
#                     c2_social_param * r2 * (global_best - swarm))
        
#         swarm += velocity

#         # Clip positions to be within bounds
#         # swarm = np.clip(swarm, bounds[:, 0], bounds[:, 1])
#         swarm = np.clip(swarm, np.array(bounds)[:, 0], np.array(bounds)[:, 1])

#         # Update personal best and global best
#         current_best_idx = np.argmin([objective_func(p) for p in swarm])
#         current_best = swarm[current_best_idx].copy()

#         for i in range(num_particles):
#             if objective_func(current_best) < objective_func(personal_best[i]):
#                 personal_best[i] = current_best.copy()

#         if objective_func(current_best) < objective_func(global_best):
#             global_best = current_best.copy()

#     return global_best, objective_func(global_best), swarm
