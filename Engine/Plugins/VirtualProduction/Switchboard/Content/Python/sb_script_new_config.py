# Copyright Epic Games, Inc. All Rights Reserved.

import json
import os
from pathlib import Path
import uuid

from switchboard.switchboard_scripting import SwitchboardScriptBase
from switchboard.switchboard_logging import LOGGER
from switchboard.config import CONFIG, SETTINGS, ROOT_CONFIGS_PATH, CONFIG_SUFFIX
from switchboard.devices.ndisplay.plugin_ndisplay import DevicenDisplay


class UnrealAutoConfigScript(SwitchboardScriptBase):

    params = {}

    def __init__(self, scriptargs):
        super().__init__(scriptargs)

        # if this causes an exception then the script won't run
        self.params = json.load(open(scriptargs))

        # delete file since it is not needed anymore
        if self.params['bAutoDeleteScriptArgsFile']:
            try:
                os.remove(scriptargs)
            except Exception as exc:
                LOGGER.warning(f"Could not delete temporary script arguments file '{scriptargs}'", exc_info=exc)

    def on_preinit(self):
        super().on_preinit()

    def on_postinit(self, app):
        super().on_postinit(app)

        # launch listener (calling this is benign if already running)
        app.listener_launcher.launch()

        # create new config

        p4_settings = {
            'p4_enabled': False,
            'source_control_workspace': None,
            'p4_sync_path': None,
            'p4_engine_path': None
        }

        config_name = self.params['configName']

        if not config_name:
            config_name = ROOT_CONFIGS_PATH / 'auto' / f'auto_{uuid.uuid4().hex}{CONFIG_SUFFIX}'

        app.create_new_config(
            file_path=config_name,
            uproject=str(Path(self.params['projectPath'])),
            engine_dir=self.params['engineDir'],
            p4_settings=p4_settings
        )

        devices = {}

        # force localhost
        if self.params['bUseLocalhost']:
            SETTINGS.ADDRESS.update_value('127.0.0.1')

        # create unreal device
        devices['Unreal'] = [
            {
                'name': f'Editor_{i+1}',
                'address': SETTINGS.ADDRESS.get_value(),
                'kwargs': {}
            }
            for i in range(self.params['numEditorDevices'])
        ]

        # create nDisplay device
        if self.params['displayClusterConfigPath']:
            DevicenDisplay.csettings['ndisplay_config_file'].update_value(self.params['displayClusterConfigPath'])
            devices['nDisplay'] = DevicenDisplay.parse_config(self.params['displayClusterConfigPath']).nodes

        # make all ips localhost
        if self.params['bUseLocalhost']:
            for devicearray in devices.values():
                for device in devicearray:
                    device['address'] = SETTINGS.ADDRESS.get_value()

        # add devices
        app.device_manager.add_devices(devices)

        # select map
        app.level = self.params['map']

        # connect Unreal and nDisplay devices to listeners
        if self.params['bAutoConnect']:
            for device in app.device_manager.devices():
                # When forcing localhost, we're usually interested in launching a single node
                if self.params['bUseLocalhost'] and isinstance(device, DevicenDisplay):
                    if device.name == DevicenDisplay.csettings['primary_device_name'].get_value():
                        device.connect_listener()
                else:
                    device.connect_listener()

        # set the project name
        CONFIG.PROJECT_NAME.update_value(f"{Path(self.params['projectPath']).stem}")

        # set the multi-user session name 
        app.set_multiuser_session_name(f"MU_{Path(config_name).stem}_1")

        # save config
        CONFIG.save()

    def on_exit(self):
        super().on_exit()
