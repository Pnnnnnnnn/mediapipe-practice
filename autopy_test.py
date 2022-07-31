import autopy
import math
import time
import random
import sys

while True:
    print(autopy.mouse.location())
    autopy.mouse.toggle(down=True)