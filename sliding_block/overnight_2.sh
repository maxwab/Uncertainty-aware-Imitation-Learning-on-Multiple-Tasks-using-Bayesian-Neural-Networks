#!/bin/bash
python Generate_BBB_Controllers.py
python Generate_GP_Controller.py -c 1 -ws 2 -po False -bc GP -tc LQR
python Generate_GP_Controller.py -c 2 -ws 2 -po False -bc GP -tc LQR
python Generate_GP_Controller.py -c 3 -ws 2 -po False -bc GP -tc LQR
python Generate_GP_Controller.py -c 4 -ws 2 -po False -bc GP -tc LQR
