using Optim, LineSearches, ForwardDiff
function msl_est_iter(initpar, lnDataShare::AbstractMatrix{T}, Delta_init::AbstractMatrix{T},
					  YL::AbstractVector{T}, YM::AbstractVector{T}, lnW::AbstractVector{T},
					  lnP::AbstractVector{T}, XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
					  XM::AbstractMatrix{T}, XF::AbstractMatrix{T}, XQ::AbstractMatrix{T},
					  ZSHK::AbstractMatrix{T}, USHK::AbstractVector{T}, wgt::AbstractVector{T},
					  sgwgt::AbstractVector{T}, nind::Int, nalt::Int, nsim::Int, dgvec::AbstractVector{Int};
					  alpha::AbstractFloat = 0.12, xdim::Int = 1, btolerance::Float64 = 1.0e-6,
					  biter::Int = 500, ftolerance::Float64 = 1.0e-10, fpiter::Int = 2000,
                      mstep::Float64 = 4.0, stepmin::Float64 = 1.0,	stepmax::Float64 = 1.0,
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
												   XT, XL, XM, XF, XQ, ZSHK, USHK, wgt,
												   nind, nalt, nsim, dgvec, alpha, xdim)
	println("\nInitial value of likelihood function = ", llk_opt_thread(coefx_old))
	algo_bt = BFGS(;alphaguess = LineSearches.InitialStatic(),
	                linesearch = LineSearches.BackTracking())

	## --- begin the Outer loop ---
	k = 1
	coefconv = one(promote_type(eltype(initpar), eltype(lnW)))
    coefx_new = copy(initpar)
    while coefconv > btolerance
        if k > biter
            printstyled("\nMaximum Iters Reached, NOT Converged!!!\n", color = :light_red)
            break
        end

        # --- BLP contraction mapping to update delta ---
        fpt_squarem!(delta_fpt, delta_new, delta_old, delta_q1, delta_q2, lnDataShare,
					 coefx_old, lnW, lnP, XT, XL, XM, XF, XQ, ZSHK, USHK, wgt, sgwgt,
					 nind, nalt, nsim, dgvec; alpha = alpha, xdim = xdim, ftolerance = ftolerance,
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
        coefconv = mreldif(coefx_new, coefx_old)
        println("The $k", "th iteration, relative coefx difference = ", "$coefconv\n")
		copyto!(coefx_old, coefx_new)
        k += 1
    end

	## TODO: calculate correct var-vcov matrix
	return(coefx_new, delta_fpt)
end
