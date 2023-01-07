import os
import sys
import unreal

# SHOTGRID_ENABLED variable will be set to "True" by the Unreal Shotgrid plugin if the environment is properly setup
shotgrid_enabled = os.getenv("UE_SHOTGRID_ENABLED", os.getenv("UE_SHOTGUN_ENABLED", "False"))

if shotgrid_enabled == "True":
    bootstrap_script = os.getenv("UE_SHOTGRID_BOOTSTRAP", os.getenv("UE_SHOTGUN_BOOTSTRAP", None))
    if bootstrap_script and os.path.exists(bootstrap_script):
        unreal.log("Shotgrid bootstrap found at {} : starting tk-unreal initialization.".format(bootstrap_script))

        # Extract path of bootstrap script and add it to the PYTHONPATH
        bootstrap_path = os.path.dirname(bootstrap_script)
        sys.path.append(os.path.abspath(bootstrap_path))

        # Now, can start the bootstrap script
        import bootstrap
    else:
        unreal.log("tk-unreal not initialized : bootstrap.py not found.")
