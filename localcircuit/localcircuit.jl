using Distributions
using LinearAlgebra
using ITensors

abstract type LocalCircuit end
const normaldist = Normal(0, 1)
# TODO: enable user to change \sigma
const weakdist = Normal(0, 0.0005)

struct LocalTensors
	tensors::Matrix{ITensor}
	idx::Index
	qb::Int
	dep::Int
	ntensor::Int
	itags::Vector{String}
	otags::Vector{String}
end

Base.size(l::LocalCircuit) = size(tensors(l))
Base.getindex(l::LocalCircuit, a::Int...) = Base.getindex(tensors(l), a...)
Base.setindex!(l::LocalCircuit, t::ITensor, a...) = 
	Base.setindex!(tensors(l), t, a...)
for f in fieldnames(LocalTensors) @eval $f(l::LocalCircuit) = l.ts.$f end

check_qd(q, d, ::Nothing) = nothing
check_qd(q, d, l::LocalCircuit) = @assert q == qb(l) && d > VQC.dep(l)

# it / ot is nothing -> default tags (input,q$(i) / output,q$(i))
# it / ot is set to Vector{String} -> input, output tags are set to them
function createlocal(::Type{T}, q, d, ii::Index, l::Union{T, Nothing}, 
					 a...; it=nothing, ot=nothing) where {T<:LocalCircuit}
	check_qd(q, d, l)
	row, col = matsize(T, q, d, a...)
	tags = Matrix{NTuple{4, String}}(undef, row, col)
	tensors = Matrix{ITensor}(undef, row, col)
	itags = (it===nothing) ? [itag(i) for i=1:nlines(T, q)] : it
	otags = (ot===nothing) ? [otag(i) for i=1:nlines(T, q)] : ot
	ts = LocalTensors(tensors, ii, q, d, ntensors(T, q, d, a...), itags, otags)
	obj = T(ts, process_args(T, a...)...)
	@assert nlines(T, q) == length(itags) && nlines(T, q) == length(otags)
	for i=1:row
		for j=1:col
			if isvalidind(T, obj, i, j)
				tags[i, j] = gettag(T, obj, i, j, itags, otags, a...)
				t::Array{Float64, 4} = getarr(T, i, j, l, a...)
				inds = (settags(ii, tg) for tg in tags[i, j])
				obj[i, j] = ITensor(t, inds...)
			end
		end
	end
	return obj
end 

createlocal(::Type{T}, q, d, a...; kw...) where T = 
	createlocal(T, q, d, Index(2), a...; kw...)
createlocal(::Type{T}, q, d, ii::Index, a...; kw...) where T = 
	createlocal(T, q, d, ii, nothing, a...; kw...)
createlocal(::Type{T}, q, d, l::T, a...; kw...) where T = 
	createlocal(T, q, d, Index(2), l, a...; kw...)

nth_tensor(l::T, i) where {T<:LocalCircuit} = 
	tensors(l)[nth_tensor_idx(T, l, i)...]

contract(l, tinit=ITensor(1)::ITensor) = 
	contract(l, 1, ntensor(l), tinit)
contract(l, e::Int, tinit=ITensor(1)::ITensor) = 
	contract(l, 1, e, tinit)
rcontract(l, tinit=ITensor(1)::ITensor) = 
	contract(l, ntensor(l), 1, tinit)
rcontract(l, e::Int, tinit=ITensor(1)::ITensor) = 
	contract(l, ntensor(l), e, tinit)

function contract(l, s, e, t)
	@assert 1 <= s && e <= ntensor(l)
	range = s <= e ? (s:e) : (s:-1:e)	
	for i=range
		t *= nth_tensor(l, i)
	end
	return t
end

fromexisting(i, j, l) = j <= dep(l)
getexisting(i, j, l) = l[i, j]
# TODO: separate identity part to another function (if necessary)
getarr(::Type{T}, i, j, l::Nothing, a...) where {T} = uniformso4()
function getarr(::Type{T}, i, j, l::T, a...) where {T}
	if fromexisting(i, j, l, a...)
		t = getexisting(i, j, l, a...).tensor
		noise_added = reshape(Vector(t.storage), 4, 4) * wnoise()
		return reshape(noise_added, 2, 2, 2, 2)
	else
		t = reshape(wnoise(), 2, 2, 2, 2)
		return permutedims(t, [1, 2, 4, 3])
	end
end
wnoise() = (A = rand(weakdist, 4, 4); exp((A - A')/2))

function uniformso4()
	vectors = [rand(normaldist, 4) for i=1:4]
	so4 = hcat(ortho(vectors...)...)
	if det(so4) < 0
		so4[:, 4] = -so4[:, 4]
	end
	reshaped = reshape(so4, 2, 2, 2, 2)
	return permutedims(reshaped, [1, 2, 4, 3]) 
end

ortho() = Vector{Float64}[]
function ortho(v::Vector{Float64}, others::Vector{Float64}...)
	result_before::Vector{Vector{Float64}} = ortho(others...)
	for vb in result_before
		v = v - dot(v, vb) * vb
	end
	return [v / norm(v), result_before...]
end

coord(i1, j1, i2, j2) = "$(i1)-$(j1),$(i2)-$(j2)"
itag(i) = "input,q$(i)"
otag(i) = "output,q$(i)"

printaddi(::Any, ::Any) = nothing
Base.show(io::IO, ::MIME"text/plain", l::T) where {T<:LocalCircuit} = 
	(print(T, ", q = $(qb(l)), d = $(dep(l))"); printaddi(T, l))

function Base.show(io::IO, l::LocalCircuit)
	Base.show(io, "text/plain", l)
	print('\n', itags(l))
	print('\n', otags(l))
end

include("LocalLadder.jl")
include("LocalBW.jl")
