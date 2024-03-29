using DataFrames
function data_prepare(df::AbstractDataFrame, lnWname::Symbol, XQJMnames::AbstractVector{Symbol},
                      XTnames::AbstractVector{Symbol}, XLnames::AbstractVector{Symbol},
                      XFnames::AbstractVector{Symbol}, XMnames::AbstractVector{Symbol},
                      XQnames::AbstractVector{Symbol}, XQparm::AbstractVector{T},
					  XQJMparm::AbstractVector{T}, LWIparm::AbstractVector{T};
					  trs::Bool = false) where T <: AbstractFloat
    ndt = nrow(df)
    city_alts = Vector{Int64}(df[:, :city_alts])
    nalt = length(unique(city_alts))
    ngvec = by(view(df, df[:, :chosen] .== 1, :), :year, nrow, sort = true)[:, 2]
    ngvec = cumsum(ngvec)
    nind = Int(ndt / nalt)
    dgvec = [locate_gidx(i, ngvec) for i = 1:nind]

    htvec = Vector{Int}(df[df[:, :chosen] .== 1, :hhtype])
	YL = Vector{Float64}(df[:, :chosen])
	YM = Vector{Float64}(df[:, :child_leftbh])

    lnW = Vector{Float64}(df[:, lnWname])
    wgtvec = Vector{Float64}(df[:w_l])
	cage9d = Vector{Int}(df[:cage9])
	sel_c9 = (cage9d .== 1) .& (YL .== 1)
	lnW = lnW .- mean(view(lnW, sel_c9), weights(view(wgtvec, sel_c9)))

    lnP = Vector{Float64}(df[:, :lnhprice])
    wgt = Vector{Float64}(df[df[:, :chosen] .== 1, :w_l])
    dage9vec = Vector{Float64}(df[df[:, :chosen] .== 1, :cage9])

    sgwgt = countmap(dgvec, weights(wgt))
    sgwgt = [sgwgt[i] for i = 1:length(sgwgt)]

	shtwgt = countmap(htvec, weights(wgt))
	shtwgt = [shtwgt[i] for i = 1:length(shtwgt)]

    swgt9 = countmap(htvec, weights(wgt .* dage9vec))[2]

    # --- initial value of delta ---
    cfreq = by(view(df, df[:, :chosen] .== 1, :), [:year, :city_alts], d -> sum(d[:, :w_l]))
    cfreq = reshape(sort(cfreq, [:year, :city_alts])[:, :x1], nalt, length(ngvec))
    cprop = cfreq ./ sum(cfreq, dims = 1)
    lnDataShare = log.(cprop)
    Delta_init = lnDataShare .- lnDataShare[1, :]'

    # --- type-specific prob. of left-behind in data ---
    # pr_lft_alt = by(view(df, df[:, :chosen] .== 1, :), [:htreat, :city_alts], sort = true) do sdf
    #     vleft = Vector{Float64}(sdf[:, :child_leftbh])
    #     wt = Vector{Float64}(sdf[:, :w_l])
    #     mean(vleft, weights(wt))
    # end
    # pr_lft_alt = reshape(pr_lft_alt[:, :x1], nalt, 2)
	#
    # pr_lft = by(view(df, df[:, :chosen] .== 1, :), :htreat, sort = true) do sdf
    #     vleft = Vector{Float64}(sdf[:, :child_leftbh])
    #     wt = Vector{Float64}(sdf[:, :w_l])
    #     mean(vleft, weights(wt))
    # end
    # pr_lft = pr_lft[:, :x1]

    # --- household preference ---
    # XTnames = [:highsch_f, :highsch_m, :age_f, :age_m, :han]
    XT = [ones(nind) convert(Array{Float64, 2}, df[df[:, :chosen] .== 1, XTnames])]

    # --- migration cost ---
    # XMnames = names(df)[(end - 17):(end - 4)]
    XM = convert(Array{Float64, 2}, df[:, XMnames])

    # --- left-behind utility loss ---
    # XLnames = [:cfemale, :nchild, :cagey, :cageysq]
    XL = convert(Array{Float64, 2}, df[df[:, :chosen] .== 1, XLnames])

    # --- fixed cost ---
    # NOTE: the most critical part of the model!!
    # XFnames = [:treat, :migscore_fcvx_city, :lnhprice, :migscore_treat, :lnhp_treat,
    #            :lnmnw_city, :nchild_lnmw, :nchild_lnhp]
    XF = [ones(ndt) convert(Array{Float64, 2}, df[:, XFnames])]

    # --- cognitive ability ---
    # XQnames = [:cfemale, :cagey, :nchild, :]
    XQ = [ones(nind) convert(Array{Float64, 2}, df[df[:, :chosen] .== 1, XQnames])]
	XQV = [ones(ndt) convert(Array{Float64, 2}, df[:, XQnames])]

    XQJ_mig = convert(Array{Float64, 2}, df[:, XQJMnames])
	XQJ_mig = XQJ_mig .- mean(view(XQJ_mig, sel_c9, :), weights(view(wgtvec, sel_c9)), dims = 1)

	nxqj = length(XQJMnames)
	lnQbar = XQV * XQparm + XQJ_mig * XQJMparm[1:nxqj] + (XQJ_mig .* YM) * XQJMparm[(nxqj+1):end]
	blft = LWIparm[1]; bw = LWIparm[2]; bitr = LWIparm[3]
	lnQbar = lnQbar + blft * YM + bw * lnW + bitr * (lnW .* YM)

    if trs == true
        XT = copy(XT')
        XL = copy(XL')
        XM = copy(XM')
        XF = copy(XF')
        XQ = copy(XQ')
        XQJ_mig = copy(XQJ_mig')
    end

    return(lnDataShare, Delta_init, YL, YM, lnW, lnP, lnQbar, XQJ_mig, wgt, sgwgt,
			shtwgt, swgt9, XT, XM, XL, XF, XQ, nalt, nind, dgvec, htvec, dage9vec)
end
