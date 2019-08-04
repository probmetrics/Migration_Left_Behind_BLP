
using Optim, ForwardDiff, LineSearches
function msm_est_iter(initpar, data_mnts::AbstractVector{T}, dwt::AbstractVector{T},
				 		lnDataShare::AbstractVector{T}, alpha::T, lnW::AbstractVector{T},
						lnP::AbstractVector{T}, XQJ_mig::AbstractVector{T},
						XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
				 		XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
						XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T},
						USHK::AbstractVector{T}, QSHK::AbstractVector{T},
						pr_lft::AbstractVector{T}, pr_lft_alt::AbstractMatrix{T},
				 		Delta_init::AbstractMatrix{T}, dgvec::AbstractVector{Int},
				 		htvec::AbstractVector{Int}, dage9vec::AbstractVector{T},
				 		wgt::AbstractVector{T}, sgwgt::AbstractVector{T},
				 		swgt9::T, nind::Int, nalt::Int,	nsim::Int; xdim::Int = 1,
						btolerance::T = 1.0e-6, biter::Int = 500,
						ftolerance::T = 1.0e-14, fpiter::Int = 2000,
                    	mstep::T = 4.0, stepmin::T = 1.0,
                    	stepmax::T = 1.0, alphaversion::Int = 3) where T <: AbstractFloat
	##
	## function to do the iterative GMM estimation
	##

	## --- containers for BLP mapping ---
	# delta_old = deepcopy(delta_init)
	DT = eltype(Delta_init)
	delta_old = deepcopy(Delta_init)
	delta_fpt = zeros(DT, size(Delta_init))
	delta_new = zeros(DT, size(Delta_init))
	delta_q1 = zeros(DT, size(Delta_init))
	delta_q2 = zeros(DT, size(Delta_init))

	## --- define GMM optim function ---
	coefx_old = copy(initpar)
	msm_opt = parm -> msm_obj(parm, data_mnts, dwt, alpha, lnW, lnP, XQJ_mig,
						 	  XT, XL, XM, XF, XQ, ZSHK, USHK, QSHK, pr_lft, pr_lft_alt,
					 	  	  delta_old, dgvec, htvec, dage9vec, wgt, sgwgt,
					 	  	  swgt9, nind, nalt, nsim; xdim = xdim)
	println("\nInitial GMM object value at fixed deltas = ", msm_opt(coefx_old))
	algo_bt = BFGS(;alphaguess = LineSearches.InitialStatic(),
	                linesearch = LineSearches.BackTracking())

	## --- begin the Outer loop ---
	k = 1
	coefconv = one(promote_type(eltype(initpar), eltype(data_mnts)))
    coefx_new = copy(initpar)
    while coefconv > btolerance
        if k > biter
            printstyled("\nMaximum Iters Reached, NOT Converged!!!\n", color = :light_red)
            break
        end

        # --- BLP contraction mapping to update delta ---
        fpt_squarem!(delta_fpt, delta_new, delta_old, delta_q1, delta_q2, lnDataShare,
					 coefx_old, lnW, lnP, XQJ_mig, XT, XL, XM, XF, XQ, ZSHK,
					 USHK, wgt, sgwgt, nind, nalt, nsim, dgvec; alpha = alpha,
					 xdim = xdim, ftolerance = ftolerance, fpiter = fpiter,
					 mstep = mstep, stepmin_init = stepmin,
					 stepmax_init = stepmax, alphaversion = alphaversion)
        copyto!(delta_old, delta_fpt)

        println("\nBegin the $k", "th GMM estimation\n")
        # --- update coefx_GMM ---
		# myftol = k <= 10 ? 1.0e-8 : 1.0e-10
		# myxtol = k <= 10 ? 1.0e-8 : 1.0e-10
		ret_msm = optimize(msm_opt, coefx_old, algo_bt,
		               	  Optim.Options(show_trace = true, iterations = 2000);
						  autodiff = :forward)
        copyto!(coefx_new, Optim.minimizer(ret_msm))
        coefconv = mreldif(coefx_new, coefx_old)
        println("The $k", "th iteration, relative coefx difference = ", "$coefconv\n")
		copyto!(coefx_old, coefx_new)
        k += 1
    end

	## TODO: calculate correct var-vcov matrix
	return(coefx_new, delta_fpt)
end
