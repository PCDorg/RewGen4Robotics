import glfw
import logging

def initialize_glfw():
    try:
        if glfw.init():
            return
        else:
            raise Exception("GLFW could not be initialized")
    except Exception as e:
        if "GLFW has already been initialized" in str(e):
            return
        else:
            logging.error(f"GLFW initialization error: {e}")
            raise