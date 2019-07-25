using DataFrames
function data_prepare(df::AbstractDataFrame, lnWname::Symbol, XQJMname::Symbol, XQJLname::Symbol,
                      XTnames::AbstractVector{Symbol}, XLnames::AbstractVector{Symbol},
                      XFnames::AbstractVector{Symbol}, XMnames::AbstractVector{Symbol},
                      XQnames::AbstractVector{Symbol}; trs::Bool = false)
    ndt = nrow(df)
    city_alts = Vector{Int64}(df[:, :city_alts])
    nalt = length(unique(city_alts))
    ngvec = by(view(df, df[:, :chosen] .== 1, :), :year, nrow, sort = true)[:, 2]
    ngvec = cumsum(ngvec)
    nind = Int(ndt / nalt)
    dgvec = [locate_gidx(i, ngvec) for i = 1:nind]

    htvec = Vector{Int}(df[df[:, :chosen] .== 1, :hhtype])

    lnW = Vector{Float64}(df[:, lnWname])
    lnW = lnW .- mean(lnW, weights(Vector{Float64}(df[:, :w_l])))
    lnP = Vector{Float64}(df[:, :lnhprice])
    XQJ_mig = Vector{Float64}(df[:, XQJMname])
    XQJ_lft = Vector{Float64}(df[:, XQJLname])
    wgt = Vector{Float64}(df[df[:, :chosen] .== 1, :w_l])
    dage9vec = Vector{Float64}(df[df[:, :chosen] .== 1, :cage9])

    sgwgt = countmap(htvec, weights(wgt))
    sgwgt = [sgwgt[i] for i = 1:length(sgwgt)]

    swgt9 = countmap(htvec, weights(wgt .* dage9vec))[2]

    # --- initial value of delta ---
    cfreq = by(view(df, df[:, :chosen] .== 1, :), [:year, :city_alts], d -> sum(d[:, :w_l]))
    cfreq = reshape(sort(cfreq, [:year, :city_alts])[:, :x1], nalt, length(ngvec))
    cprop = cfreq ./ sum(cfreq, dims = 1)
    lnDataShare = log.(cprop)
    Delta_init = lnDataShare .- lnDataShare[1, :]'

    # --- type-specific prob. of left-behind in data ---
    pr_lft_alt = by(view(df, df[:, :chosen] .== 1, :), [:htreat, :city_alts], sort = true) do sdf
        vleft = Vector{Float64}(sdf[:, :child_leftbh])
        wt = Vector{Float64}(sdf[:, :w_l])
        mean(vleft, weights(wt))
    end
    pr_lft_alt = reshape(pr_lft_alt[:, :x1], nalt, 2)

    pr_lft = by(view(df, df[:, :chosen] .== 1, :), :htreat, sort = true) do sdf
        vleft = Vector{Float64}(sdf[:, :child_leftbh])
        wt = Vector{Float64}(sdf[:, :w_l])
        mean(vleft, weights(wt))
    end
    pr_lft = pr_lft[:, :x1]

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
    # XQnames = [:cfemale, :cagey, :nchild, :tstu2_ratio]
    XQ = [ones(ndt) convert(Array{Float64, 2}, df[:, XQnames])]

    if trs == true
        XT = copy(XT')
        XL = copy(XL')
        XM = copy(XM')
        XF = copy(XF')
        XQ = copy(XQ')
    end

    return(lnDataShare, Delta_init, lnW, lnP, XQJ_mig, XQJ_lft, wgt, sgwgt, swgt9,
            XT, XM, XL, XF, XQ, pr_lft, pr_lft_alt, nalt, nind, dgvec, htvec, dage9vec)
end
