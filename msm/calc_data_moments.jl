
function mnt_var_leftbh(df::AbstractDataFrame, lnWname::Symbol,
					  	XTnames::AbstractVector{Symbol}, XLnames::AbstractVector{Symbol},
					  	XFnames::AbstractVector{Symbol}, XMnames::AbstractVector{Symbol},
					  	mnt_len::Int; nboot::Int = 200)

	leftbh_bsmnt_mat = bs_leftbh_mnts(df, lnWname, XTnames, XLnames, XFnames,
					  	  		  	  XMnames, mnt_len; nboot = nboot)
	lb_mnt_nobs = dtmnts_nobs_leftbh(df, lnWname, XTnames, XLnames, XFnames, XMnames

	lb_mnt_var = diag(cov(leftbh_bsmnt_mat, dims = 2)) .* lb_mnt_nobs
	return lb_mnt_var
end

function mnt_var_zcog(df::AbstractDataFrame, lnWname::Symbol, lnQname::Symbol,
					  QJname::Symbol, Znames::AbstractVector{Symbol},
					  XQnames::AbstractVector{Symbol}, mnt_len::Int;
					  nboot::Int = 200)
	zcog_bsmnt_mat = bs_zcog_mnts(df, lnWname, lnQname, QJname, Znames, XQnames,
				 				  mnt_len; nboot = nboot)
	zcog_mnt_nobs = dtmnts_nobs_zcog(df, lnWname, lnQname, QJname, Znames, XQnames)

	zcog_mnt_var = diag(cov(zcog_bsmnt_mat, dims = 2)) .* zcog_mnt_nobs
	return zcog_mnt_var
end

using RCall
function bs_leftbh_mnts(df::AbstractDataFrame, lnWname::Symbol,
					  XTnames::AbstractVector{Symbol}, XLnames::AbstractVector{Symbol},
					  XFnames::AbstractVector{Symbol}, XMnames::AbstractVector{Symbol},
					  mnt_len::Int; nboot::Int = 200)
	##
	## Boostrap moments for location choice and left-behind data
	##

	nd = nrow(df)

	@rput nd nboot
	R"""
	df <- data.frame(idx = 1:nd)
	set.seed(20190712)
	df_boot <- resamplr::balanced_bootstrap(df, nboot)
	boot_idx_mat <- sapply(df_boot$sample, function(x) x$idx)
	"""
	@rget boot_idx_mat

	leftbh_bs_mat = zeros(mnt_len, nboot)
	for b = 1:nboot
		bsdf = view(df, view(boot_idx_mat, :, b), :)
		view(leftbh_bs_mat, :, b) .= data_moments_leftbh(bsdf, lnWname, XTnames,
														  XLnames, XFnames, XMnames)
	end

	return leftbh_bsmnt_mat
end

using RCall
function bs_zcog_mnts(df::AbstractDataFrame, lnWname::Symbol, lnQname::Symbol,
					  QJname::Symbol, Znames::AbstractVector{Symbol},
					  XQnames::AbstractVector{Symbol}, mnt_len::Int;
					  nboot::Int = 200)
	##
	## Boostrap moments for coginitive development data
	##
	nd = nrow(df)

	@rput nd nboot
	R"""
	df <- data.frame(idx = 1:nd)
	set.seed(20190712)
	df_boot <- resamplr::balanced_bootstrap(df, nboot)
	boot_idx_mat <- sapply(df_boot$sample, function(x) x$idx)
	"""
	@rget boot_idx_mat

	zcog_bsmnt_mat = zeros(mnt_len, nboot)
	for b = 1:nboot
		bsdf = view(df, view(boot_idx_mat, :, b), :)
		view(zcog_bsmnt_mat, :, b) .= data_moments_zcog(bsdf, lnWname, lnQname,
														QJname, Znames, XQnames)
	end

	return zcog_bsmnt_mat
end

function data_moments_leftbh(df::AbstractDataFrame, lnWname::Symbol,
							 XTnames::AbstractVector{Symbol}, XLnames::AbstractVector{Symbol},
							 XFnames::AbstractVector{Symbol}, XMnames::AbstractVector{Symbol})

    leftbh = Vector{Int}(df[:child_leftbh])
	lnW = Vector{Float64}(df[lnWname])
	wgtvec = Vector{Float64}(df[:w_l])
	lnW = lnW .- mean(lnW, weights(wgtvec))

    # --- (1) type-specific left-behind probabilities in each city ---
    pr_lft_alt = by(df, [:treat, :city_alts], sort = true) do df
        vleft = Vector{Float64}(df[:child_leftbh])
        wt = Vector{Float64}(df[:w_l])
        mean(vleft, weights(wt))
    end
    pr_lft_alt = pr_lft_alt[:x1]

    # --- (2) data moments for XM ---
    XM_mnt = colwise(x -> mean(x, weights(wgtvec)), view(df, XMnames))

    # --- (3) data moments for XF and XF_lft ---
	XF_mnt = colwise(x -> mean(x, weights(wgtvec)), view(df, XFnames))
	XF_lft_mnt = colwise(x -> mean(x, weights(view(wgtvec, leftbh .== 1))),
						view(df, df[:child_leftbh] .== 1, XFnames))

    # --- (4) data moments for lnW and lnW_lft ---
	lnW_mnt = mean(lnW, weights(wgtvec))
	lnW_lft_mnt = mean(view(lnW, leftbh .== 1), weights(view(wgtvec, leftbh .== 1)))

    # --- (5) data moments for XL_lft ---
	XL_lft_mnt = colwise(x -> mean(x, weights(view(wgtvec, leftbh .== 1))),
						 view(df, df[:child_leftbh] .== 1, XLnames))

    # --- (6) data moments for XT_lft ---
	XT_lft_mnt = colwise(x -> mean(x, weights(view(wgtvec, leftbh .== 1))),
						 view(df, df[:child_leftbh] .== 1, XTnames))

	# --- (7) data moments for XT'lnW and XF XT' ---
	XT_lnW_mnt = colwise(x -> mean(x .* lnW, weights(wgtvec)), view(df, XTnames))

	XF_XT_mnt = Matrix{Float64}(view(df, XFnames))' * (Matrix{Float64}(view(df, XTnames)) .*
				wgtvec) / sum(wgtvec)
	XF_XT_mnt = vec(XF_XT_mnt)

	leftbh_mnt = vcat(pr_lft_alt, XM_mnt, XF_mnt, XF_lft_mnt, lnW_mnt, lnW_lft_mnt,
					  XL_lft_mnt, XT_lft_mnt, XT_lnW_mnt, XF_XT_mnt)
 	return leftbh_mnt
end

function data_moments_zcog(df::AbstractDataFrame, lnWname::Symbol,
						   lnQname::Symbol, QJname::Symbol,
						   Znames::AbstractVector{Symbol}, XQnames::AbstractVector{Symbol})
	##
	## NOTE: QJname should be included in XQnames
	##

	# --- moments for zshk: E(z|k) ---
	zm_mnt = by(df, :leftbh, sdf -> colwise(mean, dropmissing(sdf[Znames])), sort = true)[:x1]

	# unconditional covariance: E(z'lnw)
	tmp_df = dropmissing(df[[Znames; lnWname]], disallowmissing = true)
	zlnw_mnt = colwise(x -> mean(x .* tmp_df[lnWname]), tmp_df[Znames])

	# unconditional covariance: E(z'lnqx)
	tmp_df = dropmissing(df[[Znames; QJname]], disallowmissing = true)
	zlnqx_mnt = colwise(x -> mean(x .* tmp_df[QJname]), tmp_df[Znames])

	# --- E(lnq|k) & E(xq'lnq | k) ---
	tmp_df = dropmissing(df[[XQnames; lnQname; lnWname; :leftbh]], disallowmissing = true)

	lnq_mnt = by(tmp_df, :leftbh, sdf -> mean(sdf[lnQname]), sort = true)[:x1]
	xq_lnq_mig = colwise(x -> mean(x .* view(tmp_df, tmp_df[:leftbh] .== 0, lnQname)),
						view(tmp_df, tmp_df[:leftbh] .== 0, XQnames))
	xq_lnq_lft = colwise(x -> mean(x .* view(tmp_df, tmp_df[:leftbh] .== 1, lnQname)),
						view(tmp_df, tmp_df[:leftbh] .== 1, XQnames))

	xq_lnq_mnt = [lnq_mnt[1]; xq_lnq_mig; lnq_mnt[2]; xq_lnq_lft]

	# --- E(lnw lnq | k) ---
	lnwq_mnt = by(tmp_df, :leftbh, sdf -> mean(sdf[lnWname] .* sdf[lnQname]), sort = true)[:x1]

	# --- E(lnq^2 | k) ---
    lnq2_mnt = by(tmp_df, :leftbh, sdf -> mean(sdf[lnQname].^2), sort = true)[:x1]

	zcog_mnt = vcat(zm_mnt, zlnw_mnt, zlnqx_mnt, xq_lnq_mnt, lnwq_mnt, lnq2_mnt)
	return zcog_mnt
end

function dtmnts_nobs_leftbh(df::AbstractDataFrame, lnWname::Symbol,
							XTnames::AbstractVector{Symbol}, XLnames::AbstractVector{Symbol},
							XFnames::AbstractVector{Symbol}, XMnames::AbstractVector{Symbol})

	nXT = length(XTnames)
	nXL = length(XLnames)
	nXF = length(XFnames)
	nXM = length(XMnames)
	swt = sum(df[:w_l])
	swt_lft = sum(view(df, df[:child_leftbh] .== 1, :w_l))

    # --- (1) type-specific left-behind probabilities in each city ---
    pr_lft_alt_n = by(df, [:treat, :city_alts], x -> sum(x[:w_l]), sort = true)
    pr_lft_alt_n = pr_lft_alt_n[:x1]

    # --- (2) data moments for XM ---
    XM_mnt_n = swt * ones(nXM)

    # --- (3) data moments for XF and XF_lft ---
	XF_mnt_n = swt * ones(nXF)
	XF_lft_mnt_n = swt_lft * ones(nXF)

    # --- (4) data moments for lnW and lnW_lft ---
	lnW_mnt_n = swt
	lnW_lft_mnt_n = swt_lft

    # --- (5) data moments for XL_lft ---
	XL_lft_mnt_n = swt_lft * ones(nXL)

    # --- (6) data moments for XT_lft ---
	XT_lft_mnt_n = swt_lft * ones(nXT)

	# --- (7) data moments for XT'lnW and XF XT' ---
	XT_lnW_mnt_n = swt * ones(nXT)
	XF_XT_mnt_n = swt * ones(nXF * nXT)

	leftbh_mnt_n = vcat(pr_lft_alt_n, XM_mnt_n, XF_mnt_n, XF_lft_mnt_n, lnW_mnt_n,
					  lnW_lft_mnt_n, XL_lft_mnt_n, XT_lft_mnt_n, XT_lnW_mnt_n,
					  XF_XT_mnt_n)
 	return leftbh_mnt_n
end

function dtmnts_nobs_zcog(df::AbstractDataFrame, lnWname::Symbol,
						   lnQname::Symbol, QJname::Symbol,
						   Znames::AbstractVector{Symbol},
						   XQnames::AbstractVector{Symbol})
	##
	## NOTE: QJname should be included in XQnames
	##
	nZ = length(Znames)
	nXQ = length(XQnames)

	# --- moments for zshk: E(z|k) ---
	zm_mnt_n = by(df, :leftbh, df -> nrow(dropmissing(df[Znames])), sort = true)[:x1]
	zm_mnt_n = repeat(zm_mnt_n, inner = 2)

	# unconditional covariance: E(z'lnw)
	tmp_df = dropmissing(df[[Znames; lnWname]], disallowmissing = true)
	zlnw_mnt_n = nrow(tmp_df) * ones(nZ)

	# unconditional covariance: E(z'lnqx)
	tmp_df = dropmissing(df[[Znames; QJname]], disallowmissing = true)
	zlnqx_mnt_n = nrow(tmp_df) * ones(nZ)

	# --- E(lnq|k) & E(xq'lnq | k) ---
	tmp_df = dropmissing(df[[XQnames; lnQname; lnWname; :leftbh]], disallowmissing = true)

	lnq_mnt_n = by(tmp_df, :leftbh, df -> length(df[lnQname]), sort = true)[:x1]
	xq_lnq_mig_n = nrow(view(tmp_df, tmp_df[:leftbh] .== 0, :)) * ones(nXQ)
	xq_lnq_lft_n = nrow(view(tmp_df, tmp_df[:leftbh] .== 1, :)) * ones(nXQ)

	xq_lnq_mnt_n = [lnq_mnt_n[1]; xq_lnq_mig_n; lnq_mnt_n[2]; xq_lnq_lft_n]

	# --- E(lnw lnq | k) ---
	lnwq_mnt_n = by(tmp_df, :leftbh, df -> length(df[lnWname] .* df[lnQname]), sort = true)[:x1]

	# --- E(lnq^2 | k) ---
    lnq2_mnt_n = by(tmp_df, :leftbh, df -> length(df[lnQname]), sort = true)[:x1]

	zcog_mnt_n = vcat(zm_mnt_n, zlnw_mnt_n, zlnqx_mnt_n, xq_lnq_mnt_n,
					  lnwq_mnt_n, lnq2_mnt_n)
	return zcog_mnt_n
end
