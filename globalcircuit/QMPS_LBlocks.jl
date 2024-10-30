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

# TODO: complete these functions
function getitags(::Type{T}, L, q, d, i) where {T}

end

function getotags(::Type{T}, L, q, d, i) where {T}

end

QMPS_LBlocks{T}(a...) where {T} = createglobal(QMPS_LBlocks{T}, a...)
