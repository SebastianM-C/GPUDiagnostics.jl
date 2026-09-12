function _csv_fields(line::AbstractString)
    out = String[]
    buf = IOBuffer()
    inq = false
    i = firstindex(line)
    n = lastindex(line)
    while i <= n
        c = line[i]
        if inq
            if c == '"'
                j = nextind(line, i)
                if j <= n && line[j] == '"'
                    write(buf, '"')
                    i = j
                else
                    inq = false
                end
            else
                write(buf, c)
            end
        elseif c == '"'
            inq = true
        elseif c == ','
            push!(out, String(take!(buf)))
        else
            write(buf, c)
        end
        i = nextind(line, i)
    end
    inq && throw(ArgumentError("unterminated quoted CSV field"))
    push!(out, String(take!(buf)))
    return out
end

function _read_csv(path::AbstractString)
    header = String[]
    rows = Vector{String}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        f = _csv_fields(rstrip(line, '\r'))
        if isempty(header)
            header = f
        else
            length(f) == length(header) || throw(ArgumentError("$(basename(path)): row with $(length(f)) fields, header has $(length(header))"))
            push!(rows, f)
        end
    end
    isempty(header) && throw(ArgumentError("$(basename(path)) is empty"))
    return header, rows
end

function _csv_column(header::Vector{String}, name::AbstractString, path)
    j = findfirst(==(name), header)
    j === nothing && throw(ArgumentError("$(basename(path)): no column '$name' (columns: $(join(header, ", ")))"))
    return j
end


function _csv_get(header, row, name, default = missing)
    j = findfirst(==(name), header)
    return j === nothing || isempty(row[j]) ? default : row[j]
end
function _counter_number(s)
    ismissing(s) && return missing
    t = strip(String(s))
    lowercase(t) in ("", "n/a", "nan", "-", "not supported", "not available") && return missing
    v = tryparse(Float64, replace(t, "," => ""))
    v === nothing && throw(ArgumentError("invalid numeric counter value: $(repr(s))"))
    return _finite_metric(v)
end
function _counter_int(s)
    ismissing(s) && return missing
    n = tryparse(Int, String(s))
    n === nothing || return n
    v = _counter_number(s)
    ismissing(v) && return missing
    isinteger(v) || throw(ArgumentError("expected integer metadata, got $s"))
    return Int(v)
end
function _device_override(device, id, overrides)
    d = copy(device)
    ismissing(id) && return d
    for (k, v) in pairs(get(overrides, id, Dict()))
        d[String(k)] = v
    end
    return d
end
