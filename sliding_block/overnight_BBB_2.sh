#!/bin/bash
python Generate_BBB_Controllers.py -c 5 -ws 2 -po True
python Generate_BBB_Controllers.py -c 6 -ws 2 -po True
python Generate_BBB_Controllers.py -c 7 -ws 2 -po True
python Generate_BBB_Controllers.py -c 8 -ws 2 -po True
python Validate_BBB_Controller.py -c 5 -ws 2 -po True
python Validate_BBB_Controller.py -c 6 -ws 2 -po True
python Validate_BBB_Controller.py -c 7 -ws 2 -po True
python Validate_BBB_Controller.py -c 8 -ws 2 -po True