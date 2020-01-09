var documenterSearchIndex = {"docs":
[{"location":"data_assimilation/#Data-assimilation-1","page":"Data Assimilation","title":"Data assimilation","text":"","category":"section"},{"location":"data_assimilation/#","page":"Data Assimilation","title":"Data Assimilation","text":"Modules = [NPSMC]\nPages   = [\"data_assimilation.jl\"]\nOrder   = [:type, :function]","category":"page"},{"location":"data_assimilation/#NPSMC.DataAssimilation","page":"Data Assimilation","title":"NPSMC.DataAssimilation","text":"DataAssimilation( forecasting, method, np, xt, sigma2)\n\nparameters of the filtering method\n\nmethod :chosen method (:AnEnKF, :AnEnKS, :AnPF)\nN      : number of members (AnEnKF/AnEnKS) or particles (AnPF)\n\n\n\n\n\n","category":"type"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"EditURL = \"https://github.com/npsmc/NPSMC.jl/blob/master/examples/lorenz63.jl\"","category":"page"},{"location":"generated/lorenz63/#Lorenz-63-1","page":"Lorenz 63","title":"Lorenz 63","text":"","category":"section"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"Data assimilation are numerical methods used in geosciences to mix the information of observations (noted as y) and a dynamical model (noted as f) in order to estimate the true/hidden state of the system (noted as x) at every time step k. Usually, they are related following a nonlinear state-space model:","category":"page"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"beginaligned\nx_k  = f(x_k-1) + eta_k \ny_k  = H x_k + epsilon_k\nendaligned","category":"page"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"with eta and epsilon some independant white Gaussian noises respectively respresenting the model forecast error and the error of observation.","category":"page"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"In classical data assimilation, we require multiple runs of an explicit dynamical model f with possible severe limitations including the computational cost, the lack of consistency of the model with respect to the observed data as well as modeling uncertainties. Here, an alternative strategy is explored by developing a fully data-driven assimilation. No explicit knowledge of the dynamical model is required. Only a representative catalog of trajectories of the system is assumed to be available. Based on this catalog, the Analog Data Assimilation (AnDA) is introduced by combining machine learning with the analog method (or nearest neighbor search) and stochastic assimilation techniques including Ensemble Kalman Filter and Smoother (EnKF, EnKS) and Particle Filter (PF). We test the accuracy of the technic on different chaotic dynamical models, the Lorenz-63 and Lorenz-96 systems.  # # This Julia program is dervied from the Python library is attached to the following publication: # Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data Assimilation. Monthly Weather Review, 145(10), 4093-4107.  # If you use this library, please do not forget to cite this work.","category":"page"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"using Plots, DifferentialEquations, NPSMC","category":"page"},{"location":"generated/lorenz63/#TEST-ON-LORENZ-63-1","page":"Lorenz 63","title":"TEST ON LORENZ-63","text":"","category":"section"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"To begin, as dynamical model f, we use the Lorenz-63 chaotic system. First, we generate simulated trajectories from this dynamical model and store them into the catalog. Then, we use this catalog to emulate the dynamical model and we apply the analog data assimilation. Finally, we compare the results of this data-driven approach to the classical data assimilation (using the true Lorenz-63 equations as dynamical model.","category":"page"},{"location":"generated/lorenz63/#Generate-simulated-data-1","page":"Lorenz 63","title":"Generate simulated data","text":"","category":"section"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"σ = 10.0\nρ = 28.0\nβ = 8.0/3\n\ndt_integration = 0.01\ndt_states      = 1\ndt_obs         = 8\nparameters     = [σ, ρ, β]\nvar_obs        = [1]\nnb_loop_train  = 100\nnb_loop_test   = 10\nsigma2_catalog = 0.0\nsigma2_obs     = 2.0\n\nssm = StateSpaceModel( lorenz63,\n                       dt_integration, dt_states, dt_obs,\n                       parameters, var_obs,\n                       nb_loop_train, nb_loop_test,\n                       sigma2_catalog, sigma2_obs )","category":"page"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"compute u_0 to be in the attractor space","category":"page"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"u0    = [8.0;0.0;30.0]\ntspan = (0.0,5.0)\nprob  = ODEProblem(ssm.model, u0, tspan, parameters)\nu0    = last(solve(prob, reltol=1e-6, save_everystep=false))\n\nxt, yo, catalog = generate_data( ssm, u0 );\n\nplot( xt.t, vcat(xt.u'...)[:,1])\nscatter!( yo.t, vcat(yo.u'...)[:,1]; markersize=2)","category":"page"},{"location":"generated/lorenz63/#Classical-data-assimilation-1","page":"Lorenz 63","title":"Classical data assimilation","text":"","category":"section"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"regression = :local_linear\nsampling = :gaussian\nk, np = 100, 500\n\ndata_assimilation = DataAssimilation( ssm, xt )\nx̂_classical = data_assimilation(yo, EnKS(np), progress = false)\n@time RMSE( xt, x̂_classical)","category":"page"},{"location":"generated/lorenz63/#Analog-data-assimilation-1","page":"Lorenz 63","title":"Analog data assimilation","text":"","category":"section"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"f  = AnalogForecasting( k, xt, catalog; regression = regression, sampling = sampling )\ndata_assimilation = DataAssimilation( f, xt, ssm.sigma2_obs )\nx̂_analog = data_assimilation(yo, EnKS(np), progress = false)\n@time RMSE( xt, x̂_analog)","category":"page"},{"location":"generated/lorenz63/#Comparison-between-classical-and-analog-data-assimilation-1","page":"Lorenz 63","title":"Comparison between classical and analog data assimilation","text":"","category":"section"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"plot( xt.t, vcat(xt.u'...)[:,1], label=\"true state\")\nplot!( xt.t, vcat(x̂_classical.u'...)[:,1], label=\"classical\")\nplot!( xt.t, vcat(x̂_analog.u'...)[:,1], label=\"analog\")\nscatter!( yo.t, vcat(yo.u'...)[:,1]; markersize=2, label=\"observations\")","category":"page"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"The results show that performances of the data-driven analog data assimilation are closed to those of the model-driven data assimilation. The error can be reduced by augmenting the size of the catalog \"nblooptrain\".","category":"page"},{"location":"generated/lorenz63/#Remark-1","page":"Lorenz 63","title":"Remark","text":"","category":"section"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"Note that for all the previous experiments, we use the robust Ensemble Kalman Smoother (EnKS) with the increment or local linear regressions and the Gaussian sampling. If you want to have realistic state estimations, we preconize the use of the Particle Filter PF with the locally constant regression (regression = :locally_constant) and the multinomial sampler (sampling = :multinomial) with a large number of particles np. For more details about the different options, see the attached publication: Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data Assimilation. Monthly Weather Review, 145(10), 4093-4107.","category":"page"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"","category":"page"},{"location":"generated/lorenz63/#","page":"Lorenz 63","title":"Lorenz 63","text":"This page was generated using Literate.jl.","category":"page"},{"location":"ensemble_kalman_filters/#Ensemble-filters-1","page":"Ensemble Kalman filters","title":"Ensemble filters","text":"","category":"section"},{"location":"ensemble_kalman_filters/#","page":"Ensemble Kalman filters","title":"Ensemble Kalman filters","text":"Modules = [NPSMC]\nPages   = [\"ensemble_kalman_filters.jl\"]","category":"page"},{"location":"ensemble_kalman_filters/#NPSMC.DataAssimilation-Tuple{TimeSeries,EnKF}","page":"Ensemble Kalman filters","title":"NPSMC.DataAssimilation","text":"data_assimilation( yo, da)\n\nApply stochastic and sequential data assimilation technics using  model forecasting or analog forecasting. \n\n\n\n\n\n","category":"method"},{"location":"ensemble_kalman_filters/#NPSMC.EnKF","page":"Ensemble Kalman filters","title":"NPSMC.EnKF","text":"EnKF( np )\n\nEnsemble Kalman Filters\n\n\n\n\n\n","category":"type"},{"location":"generated/model_forecasting/#","page":"Model Forecasting","title":"Model Forecasting","text":"EditURL = \"https://github.com/npsmc/NPSMC.jl/blob/master/examples/model_forecasting.jl\"","category":"page"},{"location":"generated/model_forecasting/#Model-Forecasting-1","page":"Model Forecasting","title":"Model Forecasting","text":"","category":"section"},{"location":"generated/model_forecasting/#Set-the-State-Space-model-1","page":"Model Forecasting","title":"Set the State-Space model","text":"","category":"section"},{"location":"generated/model_forecasting/#","page":"Model Forecasting","title":"Model Forecasting","text":"using Plots\nusing NPSMC\nusing DifferentialEquations\n\nσ = 10.0\nρ = 28.0\nβ = 8.0/3\n\ndt_integration = 0.01\ndt_states      = 1\ndt_obs         = 8\nparameters     = [σ, ρ, β]\nvar_obs        = [1]\nnb_loop_train  = 100\nnb_loop_test   = 10\nsigma2_catalog = 0.0\nsigma2_obs     = 2.0\n\nssm = StateSpaceModel( lorenz63,\n                       dt_integration, dt_states, dt_obs,\n                       parameters, var_obs,\n                       nb_loop_train, nb_loop_test,\n                       sigma2_catalog, sigma2_obs )","category":"page"},{"location":"generated/model_forecasting/#","page":"Model Forecasting","title":"Model Forecasting","text":"compute u0 to be in the attractor space","category":"page"},{"location":"generated/model_forecasting/#","page":"Model Forecasting","title":"Model Forecasting","text":"u0    = [8.0;0.0;30.0]\ntspan = (0.0,5.0)\nprob  = ODEProblem(ssm.model, u0, tspan, parameters)\nu0    = last(solve(prob, reltol=1e-6, save_everystep=false))","category":"page"},{"location":"generated/model_forecasting/#Generate-data-1","page":"Model Forecasting","title":"Generate data","text":"","category":"section"},{"location":"generated/model_forecasting/#","page":"Model Forecasting","title":"Model Forecasting","text":"xt, yo, catalog = generate_data( ssm, u0 );\n\nplot( xt.t, vcat(xt.u'...)[:,1])\nscatter!( yo.t, vcat(yo.u'...)[:,1]; markersize=2)","category":"page"},{"location":"generated/model_forecasting/#Data-assimilation-with-model-forecasting-1","page":"Model Forecasting","title":"Data assimilation with model forecasting","text":"","category":"section"},{"location":"generated/model_forecasting/#","page":"Model Forecasting","title":"Model Forecasting","text":"np = 100\ndata_assimilation = DataAssimilation( ssm, xt)\n@time x̂ = data_assimilation(yo, PF(np), progress = false);\nprintln(RMSE(xt, x̂))","category":"page"},{"location":"generated/model_forecasting/#Plot-the-times-series-1","page":"Model Forecasting","title":"Plot the times series","text":"","category":"section"},{"location":"generated/model_forecasting/#","page":"Model Forecasting","title":"Model Forecasting","text":"plot(xt.t, vcat(x̂.u'...)[:,1])\nscatter!(xt.t, vcat(xt.u'...)[:,1]; markersize=2)\nplot!(xt.t, vcat(x̂.u'...)[:,2])\nscatter!(xt.t, vcat(xt.u'...)[:,2]; markersize=2)\nplot!(xt.t, vcat(x̂.u'...)[:,3])\nscatter!(xt.t, vcat(xt.u'...)[:,3]; markersize=2)","category":"page"},{"location":"generated/model_forecasting/#Plot-the-phase-space-plot-1","page":"Model Forecasting","title":"Plot the phase-space plot","text":"","category":"section"},{"location":"generated/model_forecasting/#","page":"Model Forecasting","title":"Model Forecasting","text":"p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),\n            title = \"Lorenz 63\", marker = 2)\nfor x in eachrow(vcat(x̂.u'...))\n    push!(p, x...)\nend\np","category":"page"},{"location":"generated/model_forecasting/#","page":"Model Forecasting","title":"Model Forecasting","text":"","category":"page"},{"location":"generated/model_forecasting/#","page":"Model Forecasting","title":"Model Forecasting","text":"This page was generated using Literate.jl.","category":"page"},{"location":"ideas/#Ideas-1","page":"Some ideas","title":"Ideas","text":"","category":"section"},{"location":"ideas/#Parallelization-1","page":"Some ideas","title":"Parallelization","text":"","category":"section"},{"location":"ideas/#New-Nearest-Neighbor-computation-1","page":"Some ideas","title":"New Nearest Neighbor computation","text":"","category":"section"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"Use NearestNeighborDescent","category":"page"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"The DescentGraph constructor builds the approximate kNN graph:","category":"page"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"DescentGraph(data, n_neighbors, metric; max_iters, sample_rate, precision)","category":"page"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"The k-nearest neighbors can be accessed through the indices and distances attributes. These are both KxN matrices containing ids and distances to each point's neighbors, respectively, where K = n_neighbors and N = length(data).","category":"page"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"Example:","category":"page"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"using NearestNeighborDescent\ndata = [rand(10) for _ in 1:1000] # or data = rand(10, 1000)\nn_neighbors = 5\n\n# nn descent search\ngraph = DescentGraph(data, n_neighbors)\n\n# access point i's jth nearest neighbor:\ngraph.indices[j, i]\ngraph.distances[j, i]","category":"page"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"Once constructed, the DescentGraph can be used to find the nearest neighbors to new points. This is done via the search method:","category":"page"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"search(graph, queries, n_neighbors, queue_size) -> indices, distances","category":"page"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"Example:","category":"page"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"queries = [rand(10) for _ in 1:100]\n# OR queries = rand(10, 100)\nidxs, dists = search(knngraph, queries, 4)","category":"page"},{"location":"ideas/#Questions-1","page":"Some ideas","title":"Questions","text":"","category":"section"},{"location":"ideas/#","page":"Some ideas","title":"Some ideas","text":"Revoir les notions de pas de temps / nombre d'observations","category":"page"},{"location":"particle_filters/#Particle-filters-1","page":"Particle filters","title":"Particle filters","text":"","category":"section"},{"location":"particle_filters/#","page":"Particle filters","title":"Particle filters","text":"Modules = [NPSMC]\nPages   = [\"particle_filters.jl\"]","category":"page"},{"location":"particle_filters/#NPSMC.DataAssimilation-Tuple{TimeSeries,PF}","page":"Particle filters","title":"NPSMC.DataAssimilation","text":"data_assimilation( yo, da, PF(100) )\n\nApply particle filters data assimilation technics using  model forecasting or analog forecasting. \n\n\n\n\n\n","category":"method"},{"location":"state-space/#State-Space-1","page":"State Space","title":"State Space","text":"","category":"section"},{"location":"state-space/#","page":"State Space","title":"State Space","text":"Modules = [NPSMC]\nPages   = [\"state_space.jl\"]","category":"page"},{"location":"state-space/#NPSMC.StateSpaceModel","page":"State Space","title":"NPSMC.StateSpaceModel","text":"Space-State model is defined through the following equations\n\nleft\nbeginarrayl\nX_t = m(X_t-1) + eta_t \nY_t = H(X_t) + varepsilon_t\nendarray\nright\n\nX : hidden variables\nY : observed variables\ndt_integrationis the numerical time step used to solve the ODE.\ndt_states is the number of dt_integration between X_t-1 and X_t.\ndt_obs is the number of dt_integration between Y_t-1 and Y_t.\n\n\n\n\n\n","category":"type"},{"location":"state-space/#NPSMC.SSM","page":"State Space","title":"NPSMC.SSM","text":"Generate simulated data from Space State Model\n\nvar_obs            : indices of the observed variables\ndy                 : dimension of the observations\nQ                  : model covariance\nR                  : observation covariance\ndx                 : dimension of the state\ndt_int             : fixed integration time\ndt_model           : chosen number of model time step \nvar_obs            : indices of the observed variables\ndy                 : dimension of the observations\nH                  : first and third variables are observed\nh                  : observation model\njacH               : Jacobian of the observation model(for EKS_EM only)\nQ                  : model covariance\nR                  : observation covariance\n\nExample Sinus data\n\n\n\n\n\n","category":"type"},{"location":"models/#Models-1","page":"Models","title":"Models","text":"","category":"section"},{"location":"models/#","page":"Models","title":"Models","text":"Modules = [NPSMC]\nPages   = [\"models.jl\"]","category":"page"},{"location":"models/#NPSMC.lorenz63-NTuple{4,Any}","page":"Models","title":"NPSMC.lorenz63","text":"lorenz63(du, u, p, t)\n\nLorenz-63 dynamical model u = x y z and p = sigma rho mu:\n\nfracdxdt = σ(y-x) \nfracdydt = x(ρ-z) - y \nfracdzdt = xy - βz \n\nExample Catalog\nLorenz system on wikipedia\n\n\n\n\n\n","category":"method"},{"location":"models/#NPSMC.lorenz96-NTuple{4,Any}","page":"Models","title":"NPSMC.lorenz96","text":"lorenz96(S, t, F, J)\n\nLorenz-96 dynamical model. For i=1N:\n\nfracdx_idt = (x_i+1-x_i-2)x_i-1 - x_i + F\n\nwhere it is assumed that x_-1=x_N-1x_0=x_N and x_N+1=x_1.  Here x_i is the state of the system and F is a forcing constant. \n\nLorenz 96 model on wikipedia\n\n\n\n\n\n","category":"method"},{"location":"models/#NPSMC.sinus-NTuple{4,Any}","page":"Models","title":"NPSMC.sinus","text":"sinus(du, u, p, t)\n\nSinus toy dynamical model \n\nu₁ = p₁ cos(p₁t) \n\nGenerate Sinus data\n\n\n\n\n\n","category":"method"},{"location":"ensemble_kalman_smoothers/#Ensemble-Kalman-smoothers-1","page":"Ensemble Kalman smoothers","title":"Ensemble Kalman smoothers","text":"","category":"section"},{"location":"ensemble_kalman_smoothers/#","page":"Ensemble Kalman smoothers","title":"Ensemble Kalman smoothers","text":"Modules = [NPSMC]\nPages   = [\"ensemble_kalman_smoothers.jl\"]","category":"page"},{"location":"ensemble_kalman_smoothers/#NPSMC.DataAssimilation-Tuple{TimeSeries,EnKS}","page":"Ensemble Kalman smoothers","title":"NPSMC.DataAssimilation","text":"data_assimilation( yo, da)\n\nApply stochastic and sequential data assimilation technics using  model forecasting or analog forecasting. \n\n\n\n\n\n","category":"method"},{"location":"utils/#Utilities-1","page":"Utilities","title":"Utilities","text":"","category":"section"},{"location":"utils/#","page":"Utilities","title":"Utilities","text":"Modules = [NPSMC]\nPages   = [\"utils.jl\"]","category":"page"},{"location":"utils/#NPSMC.RMSE-Tuple{Any,Any}","page":"Utilities","title":"NPSMC.RMSE","text":"RMSE(a, b)\n\nCompute the Root Mean Square Error between 2 n-dimensional vectors.\n\n\n\n\n\n","category":"method"},{"location":"utils/#NPSMC.ensure_pos_sym-Union{Tuple{Array{T,2}}, Tuple{T}} where T<:AbstractFloat","page":"Utilities","title":"NPSMC.ensure_pos_sym","text":"ensure_pos_sym(M; ϵ= 1e-8)\n\nEnsure that matrix M is positive and symmetric to avoid numerical errors when numbers are small by doing (M + M')/2 + ϵ*I\n\nreference : StateSpaceModels.jl\n\n\n\n\n\n","category":"method"},{"location":"utils/#NPSMC.inv_using_SVD-Tuple{Any,Any}","page":"Utilities","title":"NPSMC.inv_using_SVD","text":"inv_using_SVD(Mat, eigvalMax)\n\nSVD decomposition of Matrix. \n\n\n\n\n\n","category":"method"},{"location":"utils/#NPSMC.mk_stochastic!-Tuple{Array{Float64,2}}","page":"Utilities","title":"NPSMC.mk_stochastic!","text":"mk_stochastic!(w)\n\nEnsure the matrix is stochastic, i.e.,  the sum over the last dimension is 1.\n\n\n\n\n\n","category":"method"},{"location":"utils/#NPSMC.normalise!-Tuple{Any}","page":"Utilities","title":"NPSMC.normalise!","text":"normalise!( w )\n\nNormalize the entries of a multidimensional array sum to 1.\n\n\n\n\n\n","category":"method"},{"location":"utils/#NPSMC.resample!-Tuple{Array{Int64,1},Array{Float64,1}}","page":"Utilities","title":"NPSMC.resample!","text":"resample!( indx, w )\n\nMultinomial resampler.\n\n\n\n\n\n","category":"method"},{"location":"utils/#NPSMC.resample_multinomial-Tuple{Array{Float64,1}}","page":"Utilities","title":"NPSMC.resample_multinomial","text":"resample_multinomial( w )\n\nMultinomial resampler. \n\n\n\n\n\n","category":"method"},{"location":"utils/#NPSMC.sample_discrete-Tuple{Any,Any,Any}","page":"Utilities","title":"NPSMC.sample_discrete","text":"sample_discrete(prob, r, c)\n\nSampling from a non-uniform distribution. \n\n\n\n\n\n","category":"method"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"EditURL = \"https://github.com/npsmc/NPSMC.jl/blob/master/examples/sinus_data.jl\"","category":"page"},{"location":"generated/sinus_data/#Sinus-data-1","page":"Sinus data","title":"Sinus data","text":"","category":"section"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"using LinearAlgebra\nusing NPSMC\nusing Plots","category":"page"},{"location":"generated/sinus_data/#Generate-simulated-data-(Sinus-Model)-1","page":"Sinus data","title":"Generate simulated data (Sinus Model)","text":"","category":"section"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"parameters","category":"page"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"dx       = 1                # dimension of the state\ndt_int   = 1.               # fixed integration time\ndt_model = 1                # chosen number of model time step\nvar_obs  = [0]              # indices of the observed variables\ndy       = length(var_obs)  # dimension of the observations\n\nfunction h(x) # observation model\n    dx = length(x)\n    H = Matrix(I,dx,dx)\n    H .* x\nend\n\njac_h(x)  = x\nconst a = 3. :: Float64\nmx(x)     = sin.(a .* x)\njac_mx(x) = a .* cos.( a .* x)","category":"page"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"Setting covariances","category":"page"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"sig2_Q = 0.1\nsig2_R = 0.1","category":"page"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"prior state","category":"page"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"x0 = [1.]\n\nsinus3   = NPSMC.SSM(h, jac_h, mx, jac_mx, dt_int, dt_model, x0, var_obs, sig2_Q, sig2_R)","category":"page"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"generate data","category":"page"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"T        = 2000# length of the training\nX, Y     = generate_data( sinus3, x0, T)\n\nvalues = vcat(X.u'...)\nscatter( values, mx(values))\nscatter!(values[1:end-1], values[2:end], markersize=2)","category":"page"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"","category":"page"},{"location":"generated/sinus_data/#","page":"Sinus data","title":"Sinus data","text":"This page was generated using Literate.jl.","category":"page"},{"location":"generated/catalog/#","page":"Example Catalog","title":"Example Catalog","text":"EditURL = \"https://github.com/npsmc/NPSMC.jl/blob/master/examples/catalog.jl\"","category":"page"},{"location":"generated/catalog/#Example-Catalog-1","page":"Example Catalog","title":"Example Catalog","text":"","category":"section"},{"location":"generated/catalog/#","page":"Example Catalog","title":"Example Catalog","text":"using Plots, NPSMC\n\nσ = 10.0\nρ = 28.0\nβ = 8.0/3\n\ndt_integration = 0.01\ndt_states      = 1\ndt_obs         = 8\nparameters     = [σ, ρ, β]\nvar_obs        = [1]\nnb_loop_train  = 100\nnb_loop_test   = 10\nsigma2_catalog = 0.0\nsigma2_obs     = 2.0\n\nssm = StateSpaceModel( lorenz63,\n                       dt_integration, dt_states, dt_obs,\n                       parameters, var_obs,\n                       nb_loop_train, nb_loop_test,\n                       sigma2_catalog, sigma2_obs )\n\n\nxt, yo, catalog = generate_data( ssm , [10.0;0.0;0.0]);\nnothing #hide","category":"page"},{"location":"generated/catalog/#Time-series-1","page":"Example Catalog","title":"Time series","text":"","category":"section"},{"location":"generated/catalog/#","page":"Example Catalog","title":"Example Catalog","text":"plot(catalog.analogs[1,:])\nplot!(catalog.analogs[2,:])\nplot!(catalog.analogs[3,:])","category":"page"},{"location":"generated/catalog/#Phase-space-plot-1","page":"Example Catalog","title":"Phase space plot","text":"","category":"section"},{"location":"generated/catalog/#","page":"Example Catalog","title":"Example Catalog","text":"p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),\n                title = \"Lorenz 63\", marker = 1)\nfor x in eachcol(catalog.analogs)\n    push!(p, x...)\nend\np","category":"page"},{"location":"generated/catalog/#","page":"Example Catalog","title":"Example Catalog","text":"","category":"page"},{"location":"generated/catalog/#","page":"Example Catalog","title":"Example Catalog","text":"This page was generated using Literate.jl.","category":"page"},{"location":"forecasting/#Forecasting-1","page":"Forecasting","title":"Forecasting","text":"","category":"section"},{"location":"forecasting/#","page":"Forecasting","title":"Forecasting","text":"Modules = [NPSMC]\nPages   = [\"model_forecasting.jl\", \"analog_forecasting.jl\"]","category":"page"},{"location":"forecasting/#NPSMC.StateSpaceModel-Tuple{Array{Float64,2}}","page":"Forecasting","title":"NPSMC.StateSpaceModel","text":"Apply the dynamical models to generate numerical forecasts.\n\n\n\n\n\n","category":"method"},{"location":"forecasting/#NPSMC.AnalogForecasting","page":"Forecasting","title":"NPSMC.AnalogForecasting","text":"AnalogForecasting(k, xt, catalog)\n\nparameters of the analog forecasting method\n\nk            : number of analogs\nneighborhood : global analogs\ncatalog      : catalog with analogs and successors\nregression   : (:locallyconstant, :increment, :locallinear)\nsampling     : (:gaussian, :multinomial)\n\n\n\n\n\n","category":"type"},{"location":"forecasting/#NPSMC.AnalogForecasting-Tuple{Array{Float64,2}}","page":"Forecasting","title":"NPSMC.AnalogForecasting","text":"Apply the analog method on catalog of historical data \nto generate forecasts.\n\n\n\n\n\n","category":"method"},{"location":"catalog/#Catalog-1","page":"Catalog","title":"Catalog","text":"","category":"section"},{"location":"catalog/#","page":"Catalog","title":"Catalog","text":"Modules = [NPSMC]\nPages   = [\"catalog.jl\", \"generate_data.jl\"]","category":"page"},{"location":"catalog/#NPSMC.Catalog","page":"Catalog","title":"NPSMC.Catalog","text":"Catalog( data, ssm)\n\nData type to store analogs and succesors observations from the Space State model\n\nExample Catalog\n\n\n\n\n\n","category":"type"},{"location":"catalog/#NPSMC.generate_data","page":"Catalog","title":"NPSMC.generate_data","text":"generate_data( ssm, u0; seed=42)\n\nfrom StateSpace generate:\n\ntrue state (xt)\npartial/noisy observations (yo)\ncatalog\n\n\n\n\n\n","category":"function"},{"location":"catalog/#NPSMC.generate_data-Tuple{NPSMC.SSM,Array{Float64,1},Int64}","page":"Catalog","title":"NPSMC.generate_data","text":" generate_data(ssm, T_burnin, T; seed = 1)\n\nGenerate simulated data from Space State Model\n\n\n\n\n\n","category":"method"},{"location":"catalog/#NPSMC.train_test_split-Tuple{TimeSeries,TimeSeries}","page":"Catalog","title":"NPSMC.train_test_split","text":"train_test_split( X, Y; test_size)\n\nSplit time series into random train and test subsets\n\n\n\n\n\n","category":"method"},{"location":"generated/analog_forecasting/#","page":"Analog forecasting","title":"Analog forecasting","text":"EditURL = \"https://github.com/npsmc/NPSMC.jl/blob/master/examples/analog_forecasting.jl\"","category":"page"},{"location":"generated/analog_forecasting/#Analog-forecasting-1","page":"Analog forecasting","title":"Analog forecasting","text":"","category":"section"},{"location":"generated/analog_forecasting/#","page":"Analog forecasting","title":"Analog forecasting","text":"using Plots, DifferentialEquations, NPSMC\n\n\nσ = 10.0\nρ = 28.0\nβ = 8.0/3\n\ndt_integration = 0.01\ndt_states      = 1\ndt_obs         = 8\nvar_obs        = [1]\nnb_loop_train  = 100\nnb_loop_test   = 10\nsigma2_catalog = 0.0\nsigma2_obs     = 2.0\n\nssm = StateSpaceModel( lorenz63,\n                       dt_integration, dt_states, dt_obs,\n                       [σ, ρ, β], var_obs,\n                       nb_loop_train, nb_loop_test,\n                       sigma2_catalog, sigma2_obs )","category":"page"},{"location":"generated/analog_forecasting/#compute-u0-to-be-in-the-attractor-space-1","page":"Analog forecasting","title":"compute u0 to be in the attractor space","text":"","category":"section"},{"location":"generated/analog_forecasting/#","page":"Analog forecasting","title":"Analog forecasting","text":"u0    = [8.0;0.0;30.0]\ntspan = (0.0,5.0)\nprob  = ODEProblem(ssm.model, u0, tspan, ssm.params)\nu0    = last(solve(prob, reltol=1e-6, save_everystep=false))","category":"page"},{"location":"generated/analog_forecasting/#Generate-the-data-1","page":"Analog forecasting","title":"Generate the data","text":"","category":"section"},{"location":"generated/analog_forecasting/#","page":"Analog forecasting","title":"Analog forecasting","text":"xt, yo, catalog = generate_data( ssm, u0 )","category":"page"},{"location":"generated/analog_forecasting/#Create-the-forecasting-function-1","page":"Analog forecasting","title":"Create the forecasting function","text":"","category":"section"},{"location":"generated/analog_forecasting/#","page":"Analog forecasting","title":"Analog forecasting","text":"af = AnalogForecasting( 50, xt, catalog;\n    regression = :local_linear, sampling = :multinomial )","category":"page"},{"location":"generated/analog_forecasting/#Data-assimilation-1","page":"Analog forecasting","title":"Data assimilation","text":"","category":"section"},{"location":"generated/analog_forecasting/#","page":"Analog forecasting","title":"Analog forecasting","text":"np = 100\ndata_assimilation = DataAssimilation( af, xt, ssm.sigma2_obs)\nx̂ = data_assimilation(yo, EnKS(np), progress = false);\nprintln(RMSE(xt, x̂))","category":"page"},{"location":"generated/analog_forecasting/#Plot-results-1","page":"Analog forecasting","title":"Plot results","text":"","category":"section"},{"location":"generated/analog_forecasting/#","page":"Analog forecasting","title":"Analog forecasting","text":"plot(xt.t, vcat(xt.u'...)[:,1], label=:true)\nplot!(xt.t, vcat(x̂.u'...)[:,1], label=:forecasted)\nscatter!(yo.t, vcat(yo.u'...)[:,1], markersize=2, label=:observed)","category":"page"},{"location":"generated/analog_forecasting/#","page":"Analog forecasting","title":"Analog forecasting","text":"","category":"page"},{"location":"generated/analog_forecasting/#","page":"Analog forecasting","title":"Analog forecasting","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#NPSMC.jl-1","page":"Home","title":"NPSMC.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Documentation for NPSMC.jl","category":"page"},{"location":"time-series/#Time-Series-1","page":"Time Series","title":"Time Series","text":"","category":"section"},{"location":"time-series/#","page":"Time Series","title":"Time Series","text":"Modules = [NPSMC]\nPages   = [\"time_series.jl\"]","category":"page"}]
}
