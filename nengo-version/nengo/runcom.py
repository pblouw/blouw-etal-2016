r"""This modules provides access to the nengo RC settings.

Nengo RC settings will be read from the following files in the given order:
1. `INSTALL/nengo-data/nengorc`  (where INSTALL is the installation directory
    of the Nengo package)
2. A operating system specific file in the user's home directory.
   Linux: `~/.config/nengo/nengorc`
   OS X: `~/Library/Application Support/nengo/nengorc`
   Windows 7: `%userprofile%\AppData\Local\CNRGlab at UWaterloo\nengo`
3. File given in the environment variable ``$NENGORC``.
4. nengorc in the current directory.


The RC file is divided into sections by lines containing the section name
in brackets, i.e. ``[section]``. A setting is set by giving the name followed
by a ``:`` or ``=`` and the value. All lines starting with ``#`` or ``;`` are
comments.

Example
-------

This example demonstrates how to set settings in an RC file:

    [decoder_cache]
    size: 4294967296  # setting the decoder cache size to 512MiB.
"""

try:
    import configparser
except ImportError:
    import ConfigParser as configparser
import os

import nengo.utils.appdirs
import nengo.version


_APPDIRS = nengo.utils.appdirs.AppDirs(
    nengo.version.name, nengo.version.author)

DEFAULTS = {
    'decoder_cache': {
        'enabled': True,
        'readonly': False,
        'size': 4096 * 1024 * 1024,  # in bytes
        'path': os.path.join(_APPDIRS.user_cache_dir, 'decoders')
    }
}
"""The default core Nengo RC settings. Access with
``DEFAULTS[section_name][option_name]``."""

DEFAULT_RC_FILES = [
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, 'nengo-data', 'nengorc')),
    os.path.join(_APPDIRS.user_data_dir, 'nengorc'),
    os.environ['NENGORC'] if 'NENGORC' in os.environ else '',
    os.path.join(os.curdir, 'nengorc')
]
"""The RC files in the order in which they will be read."""


class _Runcom(configparser.SafeConfigParser):
    """Allows to read and write Nengo RC settings."""

    def __init__(self):
        # configparser uses old-style classes without 'super' support
        configparser.SafeConfigParser.__init__(self)
        self.reload_rc()

    def _clear(self):
        self.remove_section(configparser.DEFAULTSECT)
        for s in self.sections():
            self.remove_section(s)

    def _init_defaults(self):
        for section, settings in DEFAULTS.items():
            self.add_section(section)
            for k, v in settings.items():
                    self.set(section, k, str(v))

    def reload_rc(self, filenames=None):
        """Resets the currently loaded RC settings and loads new RC files.

        Parameters
        ----------
        filenames: iterable object
            Filenames of RC files to load.
        """
        if filenames is None:
            filenames = DEFAULT_RC_FILES

        self._clear()
        self._init_defaults()
        self.read(filenames)


runcom = _Runcom()
"""The current Nengo RC settings."""
