##
## Calculate Data moments
##

function data_moments_leftbh(LeftbhData::AbstractDataFrame, XTnames::AbstractVector{Symbol},
							 XLnames::AbstractVector{Symbol}, XFnames::AbstractVector{Symbol},
							 XMnames::AbstractVector{Symbol},
							 lnWname::Symbol, nalt::Int)

    mydf = view(LeftbhData, LeftbhData[:chosen] .== 1, :)
    choice = Vector{Int}(LeftbhData[:chosen])
    leftbh = Vector{Int}(LeftbhData[:child_leftbh])
	lnW = Vector{Float64}(LeftbhData[lnWname])
	wgtvec = Vector{Float64}(LeftbhData[:w_l])

    # --- (1) type-specific left-behind probabilities in each city ---
    pr_lft_alt = by(mydf, [:treat, :city_alts], sort = true) do df
        vleft = Vector{Float64}(df[:child_leftbh])
        wt = Vector{Float64}(df[:w_l])
        mean(vleft, weights(wt))
    end
    pr_lft_alt = pr_lft_alt[:x1]

    # --- (2) data moments for XM ---
    XM_mnt = colwise(x -> mean(x, weights(view(wgtvec, choice .== 1))), view(mydf, XMnames))

    # --- (3) data moments for XF and XF_lft ---
	XF_mnt = colwise(x -> mean(x, weights(view(wgtvec, choice .== 1))), view(mydf, XFnames))
	XF_lft_mnt = colwise(x -> mean(x, weights(view(wgtvec, (choice .== 1) .& (leftbh .== 1)))),
						view(mydf, mydf[:child_leftbh] .== 1, XFnames))

    # --- (4) data moments for lnW and lnW_lft ---
	lnW_mnt = mean(view(lnW, choice .== 1), weights(view(wgtvec, choice .== 1)))
	lnW_lft_mnt = mean(view(lnW, (choice .== 1) .& (leftbh .== 1)),
					   weights(view(wgtvec, (choice .== 1) .& (leftbh .== 1))))

    # --- (5) data moments for XL_lft ---
	XL_lft_mnt = colwise(x -> mean(x, weights(view(wgtvec, (choice .== 1) .& (leftbh .== 1)))),
						 view(mydf, mydf[:child_leftbh] .== 1, XLnames))

    # --- (6) data moments for XT_lft ---
	XT_lft_mnt = colwise(x -> mean(x, weights(view(wgtvec, (choice .== 1) .& (leftbh .== 1)))),
						 view(mydf, mydf[:child_leftbh] .== 1, XTnames))

	# --- (7) data moments for XT'lnW and XF XT' ---
	XT_lnW_mnt = colwise(x -> mean(x .* view(lnW, choice .== 1), weights(view(wgtvec, (choice .== 1)))),
						view(mydf, XTnames))

	XF_XT_mnt = Matrix{Float64}(view(mydf, XFnames))' * (Matrix{Float64}(view(mydf, XTnames)) .*
				view(wgtvec, choice .== 1)) / sum(view(wgtvec, choice .== 1))
	XF_XT_mnt = vec(XF_XT_mnt)

	leftbh_mnt = vcat(pr_lft_alt, XM_mnt, XF_mnt, XF_lft_mnt, lnW_mnt, lnW_lft_mnt,
					  XL_lft_mnt, XT_lft_mnt, XT_lnW_mnt, XF_XT_mnt)
 	return leftbh_mnt
end


function data_moments_zcog(MigBootData::AbstractDataFrame, lnWname::Symbol,
						   lnQname::Symbol, QJname::Symbol,
						   Znames::AbstractVector{Symbol}, XQnames::AbstractVector{Symbol})

	# --- moments for zshk: E(z|k) ---
	zm_mnt = by(MigBootData, :leftbh, df -> colwise(mean, dropmissing(df[Znames])), sort = true)[:x1]

	# unconditional covariance: E(z'lnw)
	tmp_df = dropmissing(MigBootData[[Znames; lnWname]], disallowmissing = true)
	zlnw_mnt = colwise(x -> mean(x .* tmp_df[lnWname]), tmp_df[Znames])

	# unconditional covariance: E(z'lnqx)
	tmp_df = dropmissing(MigBootData[[Znames; QJname]], disallowmissing = true)
	zlnqx_mnt = colwise(x -> mean(x .* tmp_df[QJname]), tmp_df[Znames])

	# --- E(lnq|k) & E(xq'lnq | k) ---
	tmp_df = dropmissing(MigBootData[[XQnames; lnQname; lnWname; :leftbh]], disallowmissing = true)

	lnq_mnt = by(tmp_df, :leftbh, df -> mean(df[lnQname]), sort = true)[:x1]
	xq_lnq_mig = colwise(x -> mean(x .* view(tmp_df, tmp_df[:leftbh] .== 0, lnQname)),
						view(tmp_df, tmp_df[:leftbh] .== 0, XQnames))
	xq_lnq_lft = colwise(x -> mean(x .* view(tmp_df, tmp_df[:leftbh] .== 1, lnQname)),
						view(tmp_df, tmp_df[:leftbh] .== 1, XQnames))

	xq_lnq_mnt = [lnq_mnt[1]; xq_lnq_mig; lnq_mnt[2]; xq_lnq_lft]

	# --- E(lnw lnq | k) ---
	lnwq_mnt = by(tmp_df, :leftbh, df -> mean(df[lnWname] .* df[lnQname]), sort = true)[:x1]

	# --- E(lnq^2 | k) ---
    lnq2_mnt = by(tmp_df, :leftbh, df -> mean(df[lnQname].^2), sort = true)[:x1]

	zcog_mnt = vcat(zm_mnt, zlnw_mnt, zlnqx_mnt, xq_lnq_mnt, lnwq_mnt, lnq2_mnt)
	return zcog_mnt
end
