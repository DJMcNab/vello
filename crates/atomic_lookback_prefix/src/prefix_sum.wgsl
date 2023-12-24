fn reduce(a: u32, b: u32) -> u32 {
    return a + b;
}

const FLAG_MASK: u32 = 0x80000000u;
const VALUE_MASK: u32 = ~FLAG_MASK;

fn remove_flag(value: u32) -> u32 {
    return value & VALUE_MASK;
}

fn with_flag(value: u32) -> u32 {
    return value | FLAG_MASK;
}

fn has_flag(value: u32) -> bool {
    return (value & FLAG_MASK) != 0u;
}

@group(0)
@binding(0)
var<storage, read_write> input: array<u32>;

// We need a specific `results` buffer, rather than reusing `input`
// to avoid a very specific data race. Namely, if the aggregate value being set
// hasn't been flushed to the cache, but
@group(0)
@binding(1)
var<storage, read_write> results: array<u32>;


@group(0)
@binding(2)
var<storage, read_write> aggregates: array<atomic<u32>>;

@group(0)
@binding(3)
var<storage, read_write> inclusive_prefices: array<atomic<u32>>;

const WG_SIZE = 64u;
const WG_SIZE_LOG2 = 6u;
var<workgroup> wg_scratch: array<u32, WG_SIZE>;
var<workgroup> loaded_value: bool;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let ix = global_id.x;
    // N.B. This is only valid on the 0th thread
    var local_agg = 0u;
    if local_id.x == 0u {
        // If a previous thread already computed our `agg`, there's no use in redoing that work
        var this_agg = atomicOr(&aggregates[workgroup_id.x], 0u);
        if has_flag(this_agg) {
            loaded_value = true;
            local_agg = remove_flag(this_agg);
        } else {
            loaded_value = false;
        }
    }
    let skip_agg = workgroupUniformLoad(&loaded_value);
    if !skip_agg {
        var agg = input[ix];
        wg_scratch[local_id.x] = agg;
        for (var i = 0u; i < WG_SIZE_LOG2; i += 1u) {
            workgroupBarrier();
            if local_id.x + (1u << i) < WG_SIZE {
                let other = wg_scratch[local_id.x + (1u << i)];
                agg = reduce(agg, other);
            }
            workgroupBarrier();
            wg_scratch[local_id.x] = agg;
        }
        if local_id.x == 0u {
            atomicStore(&aggregates[workgroup_id.x], with_flag(agg));
            local_agg = agg;
        }
    }

    // Ensure that there is a barrier before we reuse agg for the scalar fallback
    workgroupBarrier();
    // This workgroup's exclusive prefix. N.B. This only has a meaningul value on the first thread
    var exclusive_prefix = 0u;
    if workgroup_id.x == 0u {
        if local_id.x == 0u {
            atomicStore(&inclusive_prefices[0], with_flag(local_agg));
            exclusive_prefix = 0u;
        }
    } else {
        var cur_ix = workgroup_id.x - 1u;
        // Perform lookback
        loop {
            // TODO: Split this across multiple threads?
            if local_id.x == 0u {
                // We use `atomicOr` here to encourage refetching the cache?
                let prefix_ix = atomicOr(&inclusive_prefices[cur_ix], 0u);
                if has_flag(prefix_ix) {
                    exclusive_prefix = reduce(remove_flag(prefix_ix), exclusive_prefix);
                    loaded_value = true;
                } else {
                    loaded_value = false;
                }
            }
            let finished = workgroupUniformLoad(&loaded_value);
            if finished {
                break;
            }
            if local_id.x == 0u {
                let aggregate_ix = atomicOr(&aggregates[cur_ix], 0u);
                if has_flag(aggregate_ix) {
                    exclusive_prefix = reduce(remove_flag(aggregate_ix), exclusive_prefix);
                    loaded_value = true;
                } else {
                    loaded_value = false;
                }
            }
            var finished_this_iter = workgroupUniformLoad(&loaded_value);
            if !finished_this_iter {
                // Calculate aggregate for cur_ix
                let friendly_ix = cur_ix * WG_SIZE + local_id.x;
                var agg = input[friendly_ix];
                wg_scratch[local_id.x] = agg;
                for (var i = 0u; i < WG_SIZE_LOG2; i += 1u) {
                    workgroupBarrier();
                    if local_id.x + (1u << i) < WG_SIZE {
                        let other = wg_scratch[local_id.x + (1u << i)];
                        agg = reduce(agg, other);
                    }
                    workgroupBarrier();
                    wg_scratch[local_id.x] = agg;
                }
                if local_id.x == 0u {
                    atomicStore(&aggregates[cur_ix], with_flag(agg));
                    exclusive_prefix = reduce(agg, exclusive_prefix);
                }
            }
            if cur_ix == 0u {
                break;
            }
            cur_ix -= 1u;
        }
        if local_id.x == 0u {
            atomicStore(&inclusive_prefices[workgroup_id.x], with_flag(reduce(exclusive_prefix, local_agg)));
        }
    }
    results[ix] = reduce(exclusive_prefix, local_agg);
}
