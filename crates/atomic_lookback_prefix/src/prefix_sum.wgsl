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

const WG_SIZE = 256u;
const WG_SIZE_LOG2 = 8u;
const N_SEQ = 8u;
var<workgroup> wg_scratch: array<u32, WG_SIZE>;
var<workgroup> loaded_value: bool;

var<workgroup> partition_ix_var: u32;

@group(0)
@binding(4)
var<storage, read_write> partition_alloc: atomic<u32>;

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let local_ix = local_id.x;
    if local_ix == 0u {
        partition_ix_var = atomicAdd(&partition_alloc, 1u);
        loaded_value = false;
    }
    let partition_ix = workgroupUniformLoad(&partition_ix_var);

    var local = array<u32, N_SEQ>();
    let this_ix_base = (partition_ix * WG_SIZE + local_ix) * N_SEQ;
    var el = input[this_ix_base];
    // Forward scan through N_SEQ
    local[0] = el;
    for (var i = 1u; i < N_SEQ; i += 1u) {
        el = reduce(el, input[this_ix_base + i]);
        local[i] = el;
    }
    // local[0..8] is [input[this_ix_base], input[this_ix_base] + input[this_ix_base + 1], ...]
    // el is sum(input[this_ix_base..this_ix_base+7])
    let local_total = el;
    // local[0] is input[0]

    // Reverse scan through the workgroup local elements, to calculate *only* the first element
    wg_scratch[local_id.x] = el;
    for (var i = 0u; i < WG_SIZE_LOG2; i += 1u) {
        workgroupBarrier();
        if local_id.x + (1u << i) < WG_SIZE {
            let other = wg_scratch[local_id.x + (1u << i)];
            el = reduce(el, other);
        }
        workgroupBarrier();
        wg_scratch[local_id.x] = el;
    }
    if local_ix == 0u {
        atomicStore(&aggregates[partition_ix], with_flag(el));
    }
    
    // Ensure that there is a barrier before we reuse agg for the scalar fallback
    workgroupBarrier();
    // This workgroup's exclusive prefix. N.B. This only has a meaningul value on the first thread
    var exclusive_prefix = 0u;
    if partition_ix == 0u {
        if local_ix == 0u {
            atomicStore(&inclusive_prefices[0], with_flag(el));
        }
    } else {
        var cur_ix = partition_ix - 1u;
        // Perform lookback
        loop {
            // TODO: Split this across multiple threads?
            if local_ix == 0u {
                let prefix_ix = atomicLoad(&inclusive_prefices[cur_ix]);
                // let prefix_ix = atomicOr(&inclusive_prefices[cur_ix], 0u);
                let has_flag = has_flag(prefix_ix);
                if has_flag {
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
            var this_agg = 0;
            var this_idx = 0;
            loop {
                if local_ix == 0u {
                    let aggregate_ix = atomicLoad(&aggregates[cur_ix]);
                    // let aggregate_ix = atomicOr(&aggregates[cur_ix], 0u);
                    if has_flag(aggregate_ix) {
                        exclusive_prefix = reduce(remove_flag(aggregate_ix), exclusive_prefix);
                        loaded_value = true;
                    } else {
                        loaded_value = false;
                    }
                }

                let finished_this_iter = workgroupUniformLoad(&loaded_value);
                if finished_this_iter {
                    break;
                }
                // else spinlock
                // TODO: Make forward progress here
            }
            if cur_ix == 0u {
                break;
            }
            cur_ix -= 1u;
        }

        if local_ix == 0u {
            atomicStore(&inclusive_prefices[partition_ix], with_flag(reduce(exclusive_prefix, el)));
        }
    }

    // input[0]..input[local_total]
    el = local_total;
    // exclusive_prefix = 0
    if local_ix == 0u {
        el = reduce(exclusive_prefix, local_total);
    }
    // Forward scan through the next values
    wg_scratch[local_ix] = el;
    for (var i = 0u; i < WG_SIZE_LOG2; i += 1u) {
        workgroupBarrier();
        if local_id.x >= 1u << i {
            let other = wg_scratch[local_ix - (1u << i)];
            el = reduce(other, el);
        }
        workgroupBarrier();
        wg_scratch[local_id.x] = el;
    }
    workgroupBarrier();
    var res: u32;
    if local_ix == 0u {
        res = exclusive_prefix;
    } else {
        res = wg_scratch[local_ix - 1u];
    }
    for (var i = 0u ; i < N_SEQ; i += 1u) {
        results[this_ix_base + i] = reduce(res, local[i]);
    }
}
