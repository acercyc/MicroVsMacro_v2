def findX11DisplayPort():
    ''' Find localhost number for display
    1.0 - Acer 2017/02/07 15:31
    '''
    import os
    for iPort in range(10, 30):
        try:
            import matplotlib.pyplot as plt
            # print("%.1f" % iPort)
            print("localhost:%.1f" % iPort)
            break
        except:
            os.environ['DISPLAY'] = "localhost:%.1f" % iPort
