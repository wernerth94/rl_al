from multiprocessing import Pipe, Process
import tensorflow as tf
import tensorflow.keras as keras
import Agent

def child(args):
    import Agent
    a = Agent.DDVN(100, 10)
    outConn, id = args
    outConn.send(['some', i])
    outConn.close()


if __name__ == '__main__':
    tf.random.set_seed(1234)
    a = Agent.DDVN(100, 10)
    cp_callback = keras.callbacks.ModelCheckpoint('c.stateValueDir', verbose=0, save_freq=10,
                                                  save_weights_only=False)
    childs = list()
    parentConns = list()
    for i in range(5):
        parent_conn, child_conn = Pipe()
        p = Process(target=child, args=([child_conn, i],))
        childs.append(p)
        parentConns.append(parent_conn)
        p.start()
    for child, conn in zip(childs, parentConns):
        print(conn.recv())
        child.join()
