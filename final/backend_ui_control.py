import time
from multiprocessing import Process, Queue

def cpt():
    '''continuous performance task'''
    intro()
    # maybe need a input to trigger
    # may not be good in process
    
    create_bg()
    create_text()

    update_text()

    rr.log(text)
    rr.log(time)


    return key_dict

def alert(obstacle_mask):
    '''alert frame control'''
    if obstacle_mask:
        create_alert()
    else:
        destroy_alert()

def update_position(key_dict):
    while True:
        calculate_force()
        update_position()

        rr.log()

        time.sleep(0.01)



if __name__ == "__main__":
    q = Queue()
    p = Process(target=producer, args=(q,))

    p.start()
    p.join()