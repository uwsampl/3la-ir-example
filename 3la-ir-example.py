'''Example showing multiple levels of TVM IR being printed.

LSTM code by Steven.'''
import tvm
from tvm import relay
from tvm.relay.prelude import Prelude
from tvm.relay.testing.lstm import lstm_cell
import numpy as np


def generate_random_tensor(ty):
    return tvm.nd.array(
        np.random.rand(*[int(int_imm)
                         for int_imm in ty.shape]).astype(ty.dtype))


def get_lstm(batch_size, num_hidden, dtype):
    '''Returns a module where the main() function is an LSTM RNN,
    returning a tuple of two items where the first is the
    list of outputs and the second is the final hidden state'''
    mod = tvm.IRModule()
    p = Prelude(mod)
    input_type = relay.TensorType((batch_size, num_hidden), dtype)
    weight_type = relay.TensorType((4 * num_hidden, num_hidden), dtype)
    bias_type = relay.TensorType((4 * num_hidden, ), dtype)
    state_type = relay.TupleType([input_type, input_type])
    cell_type = relay.TupleType([input_type, state_type])
    state_var_type = relay.TupleType([p.l(input_type), state_type])
    input_list = relay.Var('input_list', p.l(input_type))
    init_states = relay.Var('init_states', state_type)
    cell_fn = lstm_cell(num_hidden, batch_size, dtype, "lstm_cell")
    i2h_weight = relay.Var('i2h_weight', weight_type)
    i2h_bias = relay.Var('i2h_bias', bias_type)
    h2h_weight = relay.Var('h2h_weight', weight_type)
    h2h_bias = relay.Var('h2h_bias', bias_type)
    state_var = relay.Var('state_var', state_var_type)
    input_var = relay.Var('input_var', input_type)
    cell_out = relay.Var('cell_out', cell_type)
    iteration = relay.Function([state_var, input_var],
                               relay.Let(
                                   cell_out,
                                   cell_fn(input_var,
                                           relay.TupleGetItem(state_var,
                                                              1), i2h_weight,
                                           i2h_bias, h2h_weight, h2h_bias),
                                   relay.Tuple([
                                       p.cons(relay.TupleGetItem(cell_out, 0),
                                              relay.TupleGetItem(state_var,
                                                                 0)),
                                       relay.TupleGetItem(cell_out, 1)
                                   ])), state_var_type)
    fold_res = relay.Var('fold_res', state_var_type)
    mod['rnn'] = relay.Function(
        [i2h_weight, i2h_bias, h2h_weight, h2h_bias, init_states, input_list],
        relay.Let(
            fold_res,
            p.foldl(iteration, relay.Tuple([p.nil(), init_states]),
                    input_list),
            relay.Tuple([
                p.rev(relay.TupleGetItem(fold_res, 0)),
                relay.TupleGetItem(fold_res, 1)
            ])), state_var_type)

    mod['main'] = relay.Function(
        [],
        relay.Call(mod.get_global_var('rnn'), [
            relay.const(generate_random_tensor(weight_type)),
            relay.const(generate_random_tensor(bias_type)),
            relay.const(generate_random_tensor(weight_type)),
            relay.const(generate_random_tensor(bias_type)),
            relay.Tuple([
                relay.const(generate_random_tensor(input_type)),
                relay.const(generate_random_tensor(input_type))
            ]),
            p.cons(relay.const(generate_random_tensor(input_type)), p.nil())
        ]))
    return mod


def main():
    mod = get_lstm(1, 1, 'float32')

    input_dict = {}

    USE_EXECUTOR = True
    if (USE_EXECUTOR):
        ex = relay.create_executor(mod=mod)
        out = ex.evaluate()(**input_dict)
    else:
        # TODO manually scheduling, lowering, building the code; do people still
        # do this from Relay? or does everyone just use the evaluators?
        pass


if __name__ == '__main__':
    main()
