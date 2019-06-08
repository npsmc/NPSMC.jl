var documenterSearchIndex = {"docs":
[{"location":"#NPSMC.jl-1","page":"Home","title":"NPSMC.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Documentation for NPSMC.jl","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Modules = [NPSMC]\nOrder   = [:type, :function]","category":"page"},{"location":"#NPSMC.AnalogForecasting","page":"Home","title":"NPSMC.AnalogForecasting","text":"AnalogForecasting(k, xt, catalog)\n\nparameters of the analog forecasting method\n\nk            : number of analogs\nneighborhood : global analogs\ncatalog      : catalog with analogs and successors\nregression   : (:locallyconstant, :increment, :locallinear)\nsampling     : (:gaussian, :multinomial)\n\n\n\n\n\n","category":"type"},{"location":"#NPSMC.AnalogForecasting-Tuple{Array{Float64,2}}","page":"Home","title":"NPSMC.AnalogForecasting","text":"Apply the analog method on catalog of historical data \nto generate forecasts.\n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.DataAssimilation","page":"Home","title":"NPSMC.DataAssimilation","text":"DataAssimilation( forecasting, method, np, xt, sigma2)\n\nparameters of the filtering method\n\nmethod :chosen method (:AnEnKF, :AnEnKS, :AnPF)\nN      : number of members (AnEnKF/AnEnKS) or particles (AnPF)\n\n\n\n\n\n","category":"type"},{"location":"#NPSMC.StateSpaceModel","page":"Home","title":"NPSMC.StateSpaceModel","text":"Space-State model is defined through the following equations\n\nleft\nbeginarrayl\nX_t = m(X_t-1) + eta_t \nY_t = H(X_t) + varepsilon_t\nendarray\nright\n\nX : hidden variables\nY : observed variables\ndt_integrationis the numerical time step used to solve the ODE.\ndt_states is the number of dt_integration between X_t-1 and X_t.\ndt_obs is the number of dt_integration between Y_t-1 and Y_t.\n\n\n\n\n\n","category":"type"},{"location":"#NPSMC.StateSpaceModel-Tuple{Array{Float64,2}}","page":"Home","title":"NPSMC.StateSpaceModel","text":"Apply the dynamical models to generate numerical forecasts.\n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.RMSE-Tuple{Any,Any}","page":"Home","title":"NPSMC.RMSE","text":"Compute the Root Mean Square Error between 2 n-dimensional vectors. \n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.data_assimilation-Tuple{TimeSeries,DataAssimilation}","page":"Home","title":"NPSMC.data_assimilation","text":"data_assimilation( yo, da)\n\nApply stochastic and sequential data assimilation technics using  model forecasting or analog forecasting. \n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.generate_data","page":"Home","title":"NPSMC.generate_data","text":"from StateSpace generate:\n\ntrue state (xt)\npartial/noisy observations (yo)\ncatalog\n\n\n\n\n\n","category":"function"},{"location":"#NPSMC.generate_data-Tuple{NPSMC.SSM,Array{Float64,1},Int64}","page":"Home","title":"NPSMC.generate_data","text":" generate_data(ssm, T_burnin, T; seed = 1)\n\nGenerate simulated data from Space State Model\n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.lorenz63-NTuple{4,Any}","page":"Home","title":"NPSMC.lorenz63","text":"lorenz63(du, u, p, t)\n\nLorenz-63 dynamical model \n\nbegineqnarray\nu₁(t)  =  p₁ ( u₂(t) - u₁(t)) \nu₂(t)  =  u₁ ( p₂ - u₃(t)) - u₂(t) \nu₃(t)  =  u₂(t)u₁(t) - p₃u₃(t)\nendeqnarray\n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.lorenz96-NTuple{4,Any}","page":"Home","title":"NPSMC.lorenz96","text":"lorenz96(S, t, F, J)\n\nLorenz-96 dynamical model \n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.sinus-NTuple{4,Any}","page":"Home","title":"NPSMC.sinus","text":"sinus(du, u, p, t)\n\nSinus toy dynamical model \n\nu₁ = p₁ cos(p₁t) \n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.SSM","page":"Home","title":"NPSMC.SSM","text":"Generate simulated data from Space State Model\n\nvar_obs            : indices of the observed variables\ndy                 : dimension of the observations\nQ                  : model covariance\nR                  : observation covariance\ndx                 : dimension of the state\ndt_int             : fixed integration time\ndt_model           : chosen number of model time step \nvar_obs            : indices of the observed variables\ndy                 : dimension of the observations\nH                  : first and third variables are observed\nh                  : observation model\njacH               : Jacobian of the observation model(for EKS_EM only)\nQ                  : model covariance\nR                  : observation covariance\n\n\n\n\n\n","category":"type"},{"location":"#NPSMC.inv_using_SVD-Tuple{Any,Any}","page":"Home","title":"NPSMC.inv_using_SVD","text":"inv_using_SVD(Mat, eigvalMax)\n\nSVD decomposition of Matrix. \n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.mk_stochastic!-Tuple{Array{Float64,2}}","page":"Home","title":"NPSMC.mk_stochastic!","text":"Ensure the matrix is stochastic, i.e.,  the sum over the last dimension is 1.\n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.normalise!-Tuple{Any}","page":"Home","title":"NPSMC.normalise!","text":"Normalize the entries of a multidimensional array sum to 1. \n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.resample!-Tuple{Array{Int64,1},Array{Float64,1}}","page":"Home","title":"NPSMC.resample!","text":"Multinomial resampler. \n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.resample_multinomial-Tuple{Array{Float64,1}}","page":"Home","title":"NPSMC.resample_multinomial","text":"Multinomial resampler. \n\n\n\n\n\n","category":"method"},{"location":"#NPSMC.sample_discrete-Tuple{Any,Any,Any}","page":"Home","title":"NPSMC.sample_discrete","text":"Sampling from a non-uniform distribution. \n\n\n\n\n\n","category":"method"},{"location":"example/#","page":"Example","title":"Example","text":"using Plots\nusing NPSMC\nusing DifferentialEquations","category":"page"},{"location":"example/#","page":"Example","title":"Example","text":"?StateSpaceModel","category":"page"},{"location":"example/#","page":"Example","title":"Example","text":"σ = 10.0\nρ = 28.0\nβ = 8.0/3\n\ndt_integration = 0.01\ndt_states      = 1 \ndt_obs         = 8 \nparameters     = [σ, ρ, β]\nvar_obs        = [1]\nnb_loop_train  = 100 \nnb_loop_test   = 10\nsigma2_catalog = 0.0\nsigma2_obs     = 2.0\n\nssm = StateSpaceModel( lorenz63,\n                       dt_integration, dt_states, dt_obs, \n                       parameters, var_obs,\n                       nb_loop_train, nb_loop_test,\n                       sigma2_catalog, sigma2_obs )\n\n# compute u0 to be in the attractor space\nu0    = [8.0;0.0;30.0]\ntspan = (0.0,5.0)\nprob  = ODEProblem(ssm.model, u0, tspan, parameters)\nu0    = last(solve(prob, reltol=1e-6, save_everystep=false))\n\nxt, yo, catalog = generate_data( ssm, u0 );","category":"page"},{"location":"example/#","page":"Example","title":"Example","text":"plot( xt.time, vcat(xt.values'...)[:,1])\nscatter!( yo.time, vcat(yo.values'...)[:,1]; markersize=2)","category":"page"},{"location":"example/#","page":"Example","title":"Example","text":"np = 100\nda = DataAssimilation( ssm, :EnKs, np, xt, ssm.sigma2_obs)\n@time x̂ = data_assimilation(yo, da);\nRMSE(xt, x̂)","category":"page"},{"location":"example/#","page":"Example","title":"Example","text":"plot(xt.time, vcat(x̂.values'...)[:,1])\nscatter!(xt.time, vcat(xt.values'...)[:,1]; markersize=2)\nplot!(xt.time, vcat(x̂.values'...)[:,2])\nscatter!(xt.time, vcat(xt.values'...)[:,2]; markersize=2)\nplot!(xt.time, vcat(x̂.values'...)[:,3])\nscatter!(xt.time, vcat(xt.values'...)[:,3]; markersize=2)","category":"page"},{"location":"example/#","page":"Example","title":"Example","text":"p = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),\n            title = \"Lorenz 63\", marker = 2)\nfor x in eachrow(vcat(x̂.values'...))\n    push!(p, x...)\nend\np","category":"page"}]
}
