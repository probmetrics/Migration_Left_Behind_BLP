
using Optim, ForwardDiff, LineSearches
function msm_est_nfxp(initpar, data_mnts::AbstractVector{T}, dwt::AbstractVector{T},
				 		lnDataShare::AbstractMatrix{T}, alpha::T, lnW::AbstractVector{T},
						lnP::AbstractVector{T}, XQJ_mig::AbstractMatrix{T},
						XT::AbstractMatrix{T}, XL::AbstractMatrix{T},
				 		XM::AbstractMatrix{T}, XF::AbstractMatrix{T},
						XQ::AbstractMatrix{T}, ZSHK::AbstractMatrix{T},
						USHK::AbstractVector{T}, QSHK::AbstractVector{T},
						pr_lft::AbstractVector{T}, Delta_init::AbstractMatrix{T},
				 		dgvec::AbstractVector{Int}, htvec::AbstractVector{Int},
				 		dage9vec::AbstractVector{T}, wgt::AbstractVector{T},
				 		sgwgt::AbstractVector{T}, shtwgt::AbstractVector{T},
				 		swgt9::T, nind::Int, nalt::Int,	nsim::Int; xdim::Int = 1,
						btolerance::T = 1.0e-6, biter::Int = 500,
						ftolerance::T = 1.0e-14, fpiter::Int = 2000,
                    	mstep::T = 4.0, stepmin::T = 1.0,
                    	stepmax::T = 1.0, alphaversion::Int = 3) where T <: AbstractFloat
	##
	## function to do the iterative GMM estimation
	##

	## --- define GMM optim function ---
	DT = promote_type(eltype(Delta_init), eltype(initpar))
	delta_old = deepcopy(Delta_init)
	delta_fpt = zeros(DT, size(Delta_init))
	delta_new = zeros(DT, size(Delta_init))
	delta_q1 = zeros(DT, size(Delta_init))
	delta_q2 = zeros(DT, size(Delta_init))

	## --- define GMM optim function ---
	coefx_old = copy(initpar)
	msm_opt = parm -> msm_obj(parm, data_mnts, dwt, alpha, lnW, lnP, XQJ_mig,
						 	  XT, XL, XM, XF, XQ, ZSHK, QSHK, pr_lft,
					 	  	  delta_old, dgvec, htvec, dage9vec, wgt, sgwgt, shtwgt,
					 	  	  swgt9, nind, nalt, nsim; xdim = xdim)
	msmv_old = msm_opt(coefx_old)
	println("\nInitial GMM object value at fixed deltas = ", msmv_old)
	
	algo_bt = BFGS(;alphaguess = LineSearches.InitialStatic(),
	                linesearch = LineSearches.BackTracking())

	ret_msm = optimize(msm_opt, coefx_old, algo_bt,
					   Optim.Options(show_trace = true, iterations = 3000);
					   autodiff = :forward)

	coefx = Optim.minimizer(ret_msm)
	msmv = Optim.minimum(ret_msm)

	## TODO: calculate correct var-vcov matrix
	return(coefx, delta_fpt, msmv)
end
