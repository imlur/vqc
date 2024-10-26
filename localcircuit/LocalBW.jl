struct LocalBW <: LocalCircuit
	ts::LocalTensors
	flipped::Bool
end

process_args(::Type{LocalBW}, b::Bool) = (b,)

# TODO: complete this struct (initialization, tags, ...)
