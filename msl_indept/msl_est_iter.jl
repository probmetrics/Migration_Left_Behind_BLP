using Optim, LineSearches, ForwardDiff
function msl_est_iter(initpar, lnDataShare::AbstractMatrix{T}, Delta_init::AbstractMatrix{T},
					  YL::AbstractVector{T}, YM::AbstractVector{T}, lnW::AbstractVector{T},
					  lnP::AbstractVector{T}, XbQ::AbstractVector{T}, XQJ_mig::AbstractMatrix{T},
					  XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					  XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
					  USHK::AbstractVector{T}, wgt::AbstractVector{T},
					  sgwgt::AbstractVector{T}, nind::Int, nalt::Int, nsim::Int,
					  dgvec::AbstractVector{Int}; alpha::T = 0.12, xdim::Int = 1,
					  btolerance::T = 1.0e-6, biter::Int = 500, ftolerance::T = 1.0e-14,
					  fpiter::Int = 2000, mstep::T = 4.0, stepmin::T = 1.0,	stepmax::T = 1.0,
					  alphaversion::Int = 3) where T <: AbstractFloat
	##
	## function to do the iterative GMM estimation
	##

	## --- containers for BLP mapping ---
	DT = eltype(Delta_init)
	delta_old = deepcopy(Delta_init)
	delta_fpt = zeros(DT, size(Delta_init))
	delta_new = zeros(DT, size(Delta_init))
	delta_q1 = zeros(DT, size(Delta_init))
	delta_q2 = zeros(DT, size(Delta_init))

	## --- define GMM optim function ---
	coefx_old = copy(initpar)
	llk_opt_thread = parm -> mig_leftbh_llk_thread(parm, delta_old, YL, YM, lnW, lnP,
											XbQ, XQJ_mig, XT, XL, XM, XF,
											USHK, wgt, nind, nalt, nsim, dgvec, alpha, xdim)
	llkv_old = llk_opt_thread(coefx_old)
	println("\nInitial value of likelihood function = ", llkv_old)
	algo_bt = BFGS(;alphaguess = LineSearches.InitialStatic(),
	                linesearch = LineSearches.BackTracking())

	## --- begin the Outer loop ---
	k = 1
	iter_conv = one(promote_type(eltype(initpar), eltype(lnW)))
    coefx_new = copy(initpar)
    while iter_conv > btolerance
        if k > biter
            printstyled("Maximum Iters Reached, NOT Converged!!!\n", color = :light_red)
            break
        end

        # --- BLP contraction mapping to update delta ---
        fpt_squarem!(delta_fpt, delta_new, delta_old, delta_q1, delta_q2, lnDataShare,
					 coefx_old, lnW, lnP, XbQ, XQJ_mig, XT, XL, XM, XF, 
					 USHK, wgt, sgwgt, nind, nalt, nsim, dgvec;
					 alpha = alpha, xdim = xdim, ftolerance = ftolerance,
					 fpiter = fpiter, mstep = mstep, stepmin_init = stepmin,
					 stepmax_init = stepmax, alphaversion = alphaversion)
        copyto!(delta_old, delta_fpt)

        println("\nBegin the $k", "th MSL estimation\n")
        # --- update coefx_MSL ---
		# myftol = k <= 10 ? 1.0e-8 : 1.0e-10
		# myxtol = k <= 10 ? 1.0e-8 : 1.0e-10
		ans_msl = optimize(llk_opt_thread, coefx_old, algo_bt,
		               	  Optim.Options(show_trace = true, iterations = 2000);
						  autodiff = :forward)
        copyto!(coefx_new, Optim.minimizer(ans_msl))
		llkv_new = Optim.minimum(ans_msl)

        coefconv = mreldif(coefx_new, coefx_old)
        println("The $k", "th iteration, relative coefx difference = ", "$coefconv\n")
		copyto!(coefx_old, coefx_new)

		llkconv = abs(llkv_new - llkv_old) / sqrt(abs(1.0 + llkv_old))
		println("The $k", "th iteration, log-likelihood difference = ", "$llkconv\n")
		llkv_old = llkv_new

		iter_conv = min(coefconv, llkconv)
        k += 1
    end

	## TODO: calculate correct var-vcov matrix
	return(coefx_new, delta_fpt, llkv_old)
end
