struct LocalLadder <: LocalCircuit
	ts::LocalTensors
end

function tlin(q, d, i, j, itags, otags)
	if j == 1
		if i == 1
			t1, t2 = itags[1], itags[2]
		else
			t1, t2 = coord(i, j, i-1, j), itags[i+1]
		end
	else
		if i == 1
			t1, t2 = coord(i, j, i, j-1), coord(i, j, i+1, j-1)
		else
			t1, t2 = coord(i, j, i-1, j), coord(i, j, min(i+1, q), j-1)
		end
	end
	return (t1, t2)
end

function tlout(q, d, i, j, itags, otags)
	if j == d
		if i == q
			t1, t2 = otags[q+1], otags[q]
		else
			t1, t2 = coord(i, j, i+1, j), otags[i]
		end
	else
		if i == q
			t1, t2 = coord(i, j, i, j+1), coord(i, j, i-1, j+1)
		else
			t1, t2 = coord(i, j, i+1, j), coord(i, j, max(i-1, 1), j+1)
		end
	end
	return (t1, t2)
end


LocalLadder(args...; kw...) = createlocal(LocalLadder, args...; kw...)

ntensors(::Type{LocalLadder}, q, d) = q * d
matsize(::Type{LocalLadder}, q, d) = q, d
nlines(::Type{LocalLadder}, q) = q + 1
nth_tensor_idx(::Type{LocalLadder}, l, i) = (i%qb(l)+1, div(i-1,qb(l))+1)

isvalidind(::Type{LocalLadder}, l, i, j) = true
process_args(::Type{LocalLadder}) = ()
function gettag(::Type{LocalLadder}, l, i, j,
				itags::Vector{String}, 
				otags::Vector{String}) 
	q, d = qb(l), dep(l)
	return (tlin(q, d, i, j, itags, otags)..., tlout(q, d, i, j, itags, otags)...)
end
