#!/bin/bash
python Generate_GP_Controller.py -c 3 -ws 1 -po True -bc GP -tc LQR
python Generate_GP_Controller.py -c 4 -ws 1 -po True -bc GP -tc LQR
