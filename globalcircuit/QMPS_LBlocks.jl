"""
locals : LocalCircuit objects. locals[i] is MPS for ith site
params : vector of 3-dim array, each dimensions means:
vector dim - at ith site
dim 1      - nth parameter of 
dim 2, 3   - (i, j) tensor
"""
struct QMPS_LBlocks{T<:LocalCircuit} <: QMPS{T}
	blk::QMPSBlocks{T}
end

nblocks(::Type{QMPS_LBlocks{T}}, L, q, d) where {T} = (@assert L > q + 1; L - 1)
blocksize(::Type{QMPS_LBlocks{T}}, L, q, d, i) where {T} =
	(i < L - q ? q : L - i), d
function auxargs(::Type{QMPS_LBlocks{LocalBW}}, L, q, d, i) 
	if d % 2 == 0
		return (false,)
	else
		start = L - q + 1
		return ((i - start) % 2 == 0,)
	end
end

function getitags(::Type{QMPS_LBlocks{T}}, L, q, d, i) where {T}
	# rqin : recycle qubit
	rqin = i <= L - q ? ["rqin"] : String[]
	bond = i == 1 ? "input" : "$(i-1)-$(i)" 
	input = ["bond,q$(j),$(bond)" for j=1:min(q, L - i + 1)]
	return vcat(rqin, input)
end

function getotags(::Type{QMPS_LBlocks{T}}, L, q, d, i) where {T}
	bonds = ["bond,q$(j),$(i)-$(i+1)" for j=1:min(q, L - i)]
	(i == L - 1) && (bonds[1] = "site,s$(L)")
	return vcat(bonds, ["site,s$(i)"])
end

QMPS_LBlocks{T}(a...) where {T} = createglobal(QMPS_LBlocks{T}, a...)

function prepnextcont(::Type{QMPS_LBlocks{T}}, g, t, i) where {T}
	if i < g.blk.L - g.blk.q
		ind, nextblk = g.blk.i, g.blk.locals[i+1]
		tag2add = nextblk.ts.itags[1]
		return t * ITensor([1, 0], settags(ind, tag2add))
	end
	return t
end
