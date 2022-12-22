# Copyright Epic Games, Inc. All Rights Reserved.

from _unreal_core import *
import sys as _sys
import warnings as _warnings
import os as _os
import ssl as _ssl
import platform as _platform

class _Logger(object):
	def __init__(self, log_func, flush_func):
		self.encoding = "utf-8"
		self.log_func = log_func
		self.flush_func = flush_func
	def write(self, log_text):
		self.log_func(log_text)
	def writelines(self, lines):
		for line in lines:
			self.write(line)
	def flush(self):
		self.flush_func()

_sys.stdout = _Logger(log, log_flush)
_sys.stderr = _Logger(log_error, log_flush)

def _redirect_warning(message, category, filename, lineno, file=None, line=None):
	log_warning(_warnings.formatwarning(message, category, filename, lineno))
_warnings.showwarning = _redirect_warning

# On Mac/Linux, the Python and SSL libraries are built by Epic. Normally, when we run 'make install' for the SSL
# library, the installer copies the default certificates (.pem file) in the install folder where SSL expects them. But
# in UE, we copy the SSL lib and the certificates are not 'installed' where SSL expects them. If the user doesn't configure
# where to find the SSL certificates, Python will try to find the SSL default location and will likely fail. Python can use
# the SSL certificates from the 'certifi' package, which are from Mozilla. It turns out that UE also ships the '.pem' coming
# from Mozilla in Engine\Content\Certificates\ThirdParty\cacert.pem. As a convenience for users, we set up Python to use the
# certificates distributed with the Engine. Note that if the default certificates UE provides are not adequate, the user can
# set SSL_CERT_FILE enviroment variable or directly set the ones to use when calling 'ssl.create_default_context(...)'
if _platform.system() == 'Darwin' or _platform.system() == 'Linux':
	# If the default path to the certificates doesn't exist and user didn't configure the corresponding environment variable.
	cert_file_env = 'SSL_CERT_FILE'
	if not _os.path.exists(_ssl.get_default_verify_paths().openssl_cafile) and cert_file_env not in _os.environ:
		script_dir = _os.path.dirname(_os.path.abspath(__file__))
		rel_path = _os.path.join(script_dir, '..', '..', '..', '..', '..', 'Content', 'Certificates', 'ThirdParty', 'cacert.pem')
		_os.environ[cert_file_env] = _os.path.normpath(rel_path)

def uclass():
	'''decorator used to define UClass types from Python'''
	def _uclass(type):
		generate_class(type)
		return type
	return _uclass
	
def ustruct():
	'''decorator used to define UStruct types from Python'''
	def _ustruct(type):
		generate_struct(type)
		return type
	return _ustruct
	
def uenum():
	'''decorator used to define UEnum types from Python'''
	def _uenum(type):
		generate_enum(type)
		return type
	return _uenum

def uvalue(val, meta=None):
	'''function used to define constant values from Python'''
	return ValueDef(val, meta)
	
def uproperty(type, meta=None, getter=None, setter=None):
	'''function used to define UProperty fields from Python'''
	return PropertyDef(type, meta, getter, setter)

def ufunction(meta=None, ret=None, params=None, override=None, static=None, pure=None, getter=None, setter=None):
	'''decorator used to define UFunction fields from Python'''
	def _ufunction(func):
		return FunctionDef(func, meta, ret, params, override, static, pure, getter, setter)
	return _ufunction
	
