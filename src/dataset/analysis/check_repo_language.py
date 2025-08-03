import os
import json
import requests
import time

from tqdm import tqdm
from dotenv import load_dotenv
from animate import Loader
from urllib.parse import ParseResult, urlparse, quote


# <---- LOAD ENVIRONMENT VARIABLES FROM THE .env FILE ---->
load_dotenv()

# <---- AUTH FIELDS ---->
GITHUB_TOKEN = os.environ.get("GITHUB_PAT")
GITLAB_TOKEN = os.environ.get("GITLAB_PAT")

GITHUB_HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
GITLAB_HEADERS = {"Private-Token": f"{GITLAB_TOKEN}"}


# The list of projects you provided
# PROJECTS: set[str] = {
#     "glibc",
#     "incubator-doris",
#     "pam_pkcs11",
#     "jpeg-quantsmooth",
#     "openldap",
#     "veyon",
#     "libyaml",
#     "usbredir",
#     "bitlbee",
#     "doas",
#     "util-linux",
#     "libsass",
#     "gnulib",
#     "cockpit",
#     "shapelib",
#     "mongoose-os",
#     "libqb",
#     "openrazer",
#     "mysql-wsrep",
#     "ssh2",
#     "uWebSockets",
#     "opencv",
#     "staging",
#     "libplist",
#     "iodine",
#     "OpenSC",
#     "gdm",
#     "sparc",
#     "atheme",
#     "openssh-portable",
#     "pycrypto",
#     "shotcut",
#     "deepin-clone",
#     "gst-plugins-base",
#     "haproxy",
#     "abrt",
#     "eucalyptus",
#     "3proxy",
#     "libssh",
#     "PackageKit",
#     "e2fsprogs",
#     "botan",
#     "libgxps",
#     "nfdump",
#     "libevent",
#     "tar",
#     "texlive-source",
#     "pango",
#     "rpm",
#     "lz4",
#     "xrdp",
#     "pykerberos",
#     "qBittorrent",
#     "ruby",
#     "cyrus-sasl",
#     "libzip",
#     "osquery",
#     "zstd",
#     "ompl",
#     "taglib",
#     "swtpm",
#     "clamav-devel",
#     "gstreamer",
#     "thunar",
#     "bro",
#     "pure-ftpd",
#     "mongo-c-driver-legacy",
#     "linux",
#     "char-misc",
#     "dbus",
#     "quagga",
#     "aide",
#     "proxychains-ng",
#     "md4c",
#     "tpm2.0-tools",
#     "davs2",
#     "pam_tacplus",
#     "libextractor",
#     "graphite",
#     "ecdsautils",
#     "nDPI",
#     "open62541",
#     "wireshark",
#     "mod_auth_openidc",
#     "neomutt",
#     "pesign",
#     "libwebm",
#     "netmask",
#     "tk",
#     "libcaca",
#     "libheif",
#     "protobuf",
#     "libslirp",
#     "samurai",
#     "openmpt",
#     "freetype2-demos",
#     "mvfst",
#     "ntfs-3g",
#     "wesnoth",
#     "spice",
#     "tor",
#     "media_tree",
#     "lldpd",
#     "icu",
#     "libxml2",
#     "jerryscript",
#     "Sigil",
#     "gnome-font-viewer",
#     "NetHack",
#     "psutil",
#     "mailutils",
#     "librsvg",
#     "u-boot",
#     "nautilus",
#     "NetworkManager-vpnc",
#     "p11-kit",
#     "grub",
#     "thrift",
#     "FreeRTOS-Kernel",
#     "jhead",
#     "trojita",
#     "wildmidi",
#     "ncurses",
#     "krecipes",
#     "pupnp",
#     "wireless",
#     "mintty",
#     "proftpd",
#     "librelp",
#     "libsndfile",
#     "leptonica",
#     "nagioscore",
#     "Onigmo",
#     "net-next",
#     "libbsd",
#     "htslib",
#     "dcmtk",
#     "polkit",
#     "yodl",
#     "gnome-shell",
#     "booth",
#     "colord",
#     "puma",
#     "selinux",
#     "gdal",
#     "mariadb-connector-c",
#     "libgsf",
#     "gnash",
#     "cxxtools",
#     "ippusbxd",
#     "iniparser",
#     "ekiga",
#     "tensorflow",
#     "libinfinity",
#     "enlightenment",
#     "v4l2loopback",
#     "resiprocate",
#     "moddable",
#     "amanda",
#     "libsixel",
#     "libgcrypt",
#     "dpdk-stable",
#     "astc-encoder",
#     "tinyexr",
#     "zsh",
#     "ntp",
#     "file-roller",
#     "node",
#     "envoy",
#     "fribidi",
#     "subversion",
#     "gnuplot",
#     "njs",
#     "CImg",
#     "roundcubemail",
#     "tty",
#     "cyrus-imapd",
#     "cluster-glue",
#     "libusbmuxd",
#     "quassel",
#     "iperf",
#     "profanity",
#     "rabbitmq-c",
#     "radare2",
#     "exif",
#     "mod_wsgi",
#     "ok-file-formats",
#     "bubblewrap",
#     "libtool",
#     "vim",
#     "libcdio",
#     "mosquitto",
#     "libx11",
#     "detect-character-encoding",
#     "arangodb",
#     "libmicrohttpd",
#     "wolfssl",
#     "evolution-ews",
#     "libjpeg",
#     "SoftHSMv2",
#     "acpica",
#     "menu-cache",
#     "Varnish-Cache",
#     "fbthrift",
#     "user-namespace",
#     "flatpak",
#     "gsasl",
#     "tcmu-runner",
#     "cronie",
#     "das_watchdog",
#     "libxkbcommon",
#     "kernel",
#     "pacemaker",
#     "pgbouncer",
#     "ffjpeg",
#     "libxmljs",
#     "libpcap",
#     "gst-plugins-ugly",
#     "advancecomp",
#     "LuaJIT",
#     "Espruino",
#     "nokogiri",
#     "acrn-hypervisor",
#     "ocaml",
#     "furnace",
#     "systemd-stable",
#     "cifs-utils",
#     "tidy-html5",
#     "mesos",
#     "jabberd2",
#     "exim",
#     "sddm",
#     "pycurl",
#     "cgit",
#     "kvm-guest-drivers-windows",
#     "minetest",
#     "samba",
#     "ovs",
#     "jbig2dec",
#     "unicorn",
#     "suricata",
#     "libming",
#     "Fast-DDS",
#     "VeraCrypt",
#     "postgres",
#     "lynx-snapshots",
#     "turbovnc",
#     "bareos",
#     "kdeconnect-kde",
#     "mujs",
#     "guile",
#     "quazip",
#     "libexpat",
#     "faad2",
#     "fontforge",
#     "Singular",
#     "linux-next",
#     "linux-stable-rt",
#     "Little-CMS",
#     "at-spi2-atk",
#     "capnproto",
#     "drogon",
#     "c-ares",
#     "exo",
#     "imagick",
#     "graphviz",
#     "libjxl",
#     "openssl",
#     "monkey",
#     "agoo",
#     "ipmitool",
#     "linux-pam",
#     "vte",
#     "edk2",
#     "nspluginwrapper",
#     "ModSecurity",
#     "percona-xtrabackup",
#     "admesh",
#     "opus",
#     "ettercap",
#     "eog",
#     "ntopng",
#     "bdwgc",
#     "upx",
#     "mongoose",
#     "sudo",
#     "engine",
#     "krb5",
#     "gssproxy",
#     "hhvm",
#     "cantata",
#     "memcached",
#     "linux-2.6",
#     "yara",
#     "pngquant",
#     "tpm2-tools",
#     "miniupnp",
#     "perl5",
#     "bwa",
#     "openjpeg",
#     "libmysofa",
#     "qemu-kvm",
#     "gvfs",
#     "lepton",
#     "libfep",
#     "frr",
#     "AFFLIBv3",
#     "cairo",
#     "smb4k",
#     "ioq3",
#     "mongo",
#     "qpid-proton",
#     "php-src",
#     "shadowsocks-libev",
#     "openvpn",
#     "cpython",
#     "hermes",
#     "htcondor",
#     "libdwarf-code",
#     "zcash",
#     "LIEF",
#     "libredwg",
#     "subconverter",
#     "tinygltf",
#     "gpmf-parser",
#     "389-ds-base",
#     "unbound",
#     "mruby",
#     "gnumeric",
#     "dash",
#     "libbson",
#     "cortx-s3server",
#     "bird",
#     "mupdf",
#     "passenger",
#     "docker-credential-helpers",
#     "libmspack",
#     "open5gs",
#     "poco",
#     "http-parser",
#     "kamailio",
#     "AESCrypt",
#     "linux-fs",
#     "network-manager-applet",
#     "kvm",
#     "linux-stable",
#     "augeas",
#     "history",
#     "capstone",
#     "libreswan",
#     "sgminer",
#     "libfuse",
#     "libssh2",
#     "glib",
#     "open-vm-tools",
#     "stunnel",
#     "node-sqlite3",
#     "slurm",
#     "libreport",
#     "c-shquote",
#     "portable",
#     "gnome-screensaver",
#     "libcomps",
#     "tinyproxy",
#     "date",
#     "totd",
#     "brotli",
#     "stb",
#     "suhosin",
#     "librepo",
#     "xrootd",
#     "libjpeg-turbo",
#     "snapd",
#     "kitty",
#     "pdfresurrect",
#     "libnbd",
#     "re2c",
#     "proxygen",
#     "fdkaac",
#     "libmatroska",
#     "cpp-peglib",
#     "relic",
#     "curl",
#     "flac",
#     "WavPack",
#     "qemu",
#     "gnutls",
#     "liblouis",
#     "opa-fm",
#     "raptor",
#     "cimg",
#     "ext-http",
#     "shadow",
#     "jdk17u",
#     "varnish-cache",
#     "libtpms",
#     "typed_ast",
#     "nghttp2",
#     "lua",
#     "squid",
#     "libsolv",
#     "Crow",
#     "mongo-c-driver",
#     "LibRaw-demosaic-pack-GPL2",
#     "tsMuxer",
#     "matio",
#     "openenclave",
#     "net-snmp",
#     "qtbase",
#     "tmux",
#     "shellinabox",
#     "radvd",
#     "libgdata",
#     "udisks",
#     "gpac",
#     "libndp",
#     "axtls-8266",
#     "OpenDoas",
#     "gcc",
#     "openscad",
#     "tpm",
#     "doom-vanille",
#     "t1utils",
#     "zziplib",
#     "bzip2",
#     "pigeonhole",
#     "haproxy-1.4",
#     "gnome-bluetooth",
#     "rizin",
#     "libvncserver",
#     "keepkey-firmware",
#     "isolated-vm",
#     "atomicparsley",
#     "bwm-ng",
#     "libvirt",
#     "accel-ppp",
#     "tcpdump",
#     "systemd",
#     "libcroco",
#     "busybox",
#     "libseccomp",
#     "collectd",
#     "LibRaw",
#     "123elf",
#     "ceph",
#     "ark",
#     "libuv",
#     "owntone-server",
#     "ndjbdns",
#     "nf",
#     "webcc",
#     "poppler",
#     "lighttpd1.4",
#     "runc",
#     "swoole-src",
#     "bitcoin",
#     "uftpd",
#     "qpdf",
#     "wireless-drivers",
#     "autotrace",
#     "uriparser",
#     "aspell",
#     "torque",
#     "bluez",
#     "gerbv",
#     "libxsmm",
#     "icecast-server",
#     "tigervnc",
#     "lwip",
#     "libmodbus",
#     "mod_auth_mellon",
#     "knc",
#     "zephyr",
#     "tip",
#     "openfortivpn",
#     "shim",
#     "clutter",
#     "weechat",
#     "mksh",
#     "firejail",
#     "nbd",
#     "ppp",
#     "uwsgi",
#     "sfntly",
#     "rsyslog",
#     "NetworkManager",
#     "tang",
#     "zfs",
#     "feh",
#     "gzip",
#     "unixODBC",
#     "cJSON",
#     "ghostpdl",
#     "pupnp-code",
#     "nasm",
#     "gst-plugins-bad",
#     "soundtouch",
#     "audiofile",
#     "evolution-data-server",
#     "ArduinoJson",
#     "ibus",
#     "libvpx",
#     "opensips",
#     "mysql-server",
#     "DBD-mysql",
#     "libidn2",
#     "ZLMediaKit",
#     "php-radius",
#     "ytnef",
#     "libssh-mirror",
#     "cracklib",
#     "coreboot",
#     "libidn",
#     "libpod",
#     "zlib",
#     "vino",
#     "nanopb",
#     "sysstat",
#     "media-tree",
#     "rawstudio",
#     "quagga-RE",
#     "cups",
#     "FreeRDP",
#     "GIMP",
#     "spice-vd_agent",
#     "corosync",
#     "wayland",
#     "mumble",
#     "mutt",
#     "keepalived",
#     "ImageMagick",
#     "x11vnc",
#     "libhtp",
#     "libass",
#     "xorg-xserver",
#     "seatd",
#     "mosh",
#     "gegl",
#     "inspircd",
#     "udev",
#     "libiec61850",
#     "android-gif-drawable",
#     "wolfssh",
#     "at91bootstrap",
#     "pigz",
#     "linux-fbdev",
#     "Bento4",
#     "yajl-ruby",
#     "FFmpeg",
#     "at-spi2-core",
#     "crun",
#     "UltraVNC",
#     "cgal",
#     "hexchat",
#     "uclibc-ng",
#     "libxslt",
#     "chrony",
#     "irssi-proxy",
#     "openthread",
#     "ZRTPCPP",
#     "redcarpet",
#     "libsoup",
#     "iproute2",
#     "Openswan",
#     "gpsd",
#     "libffi",
#     "openj9",
#     "fastd",
#     "optee_os",
#     "mbedtls",
#     "ImageMagick6",
#     "pcre2",
#     "openexr",
#     "android_security",
#     "LoRaMac-node",
#     "json",
#     "muon",
#     "squirrel",
#     "cinnamon-screensaver",
#     "hivex",
#     "gnupg",
#     "jdk11u",
#     "firebird",
#     "fastecdsa",
#     "winsparkle",
#     "SDL",
#     "percona-xtradb-cluster",
#     "libtiff",
#     "MonetDB",
#     "libu2f-host",
#     "bluetooth-next",
#     "ldns",
#     "gst-plugins-good",
#     "trafficserver",
#     "gnome-session",
#     "SDL_ttf",
#     "evolution",
#     "dpdk",
#     "bzrtp",
#     "libarchive",
#     "bash",
#     "target-pending",
#     "webserver",
#     "mbed-os",
#     "cgminer",
#     "fizz",
#     "lsquic",
#     "libinput",
#     "empathy",
#     "lftp",
#     "git",
#     "znc",
#     "json-c",
#     "pound",
#     "libspiro",
#     "pdns",
#     "ubridge",
#     "nefarious2",
#     "heimdal",
#     "csync2",
#     "httpd",
#     "Pillow",
#     "glewlwyd",
#     "gdk-pixbuf",
#     "tflite-micro",
#     "m4",
#     "newsbeuter",
#     "cryptopp",
#     "android_kernel_xiaomi_msm8996",
#     "lasso",
#     "MilkyTracker",
#     "exiv2",
#     "akashi",
#     "picocom",
#     "polipo",
#     "scylladb",
#     "rawspeed",
#     "avahi",
#     "jasper",
#     "aircrack-ng",
#     "nmap",
#     "libdxfrw",
#     "c-blosc2",
#     "xterm-snapshots",
#     "libmobi",
#     "file",
#     "wangle",
#     "virglrenderer",
#     "pyfribidi",
#     "asylo",
#     "ast",
#     "harfbuzz",
#     "pjproject",
#     "crawl",
#     "redis",
#     "sleuthkit",
#     "iipsrv",
#     "sound",
#     "h2o",
#     "netdata",
#     "yubico-pam",
#     "glib-networking",
#     "sssd",
#     "libpng",
#     "ssdp-responder",
#     "kde-cli-tools",
#     "libsrtp",
#     "jdk11u-dev",
#     "lxc",
#     "libtomcrypt",
#     "polarssl",
#     "vorbis",
#     "tinc",
#     "qcad",
#     "nf-next",
#     "aubio",
#     "mod_h2",
#     "nbdkit",
#     "teeworlds",
#     "libexif",
#     "wget",
#     "net",
#     "spice-common",
#     "freeipa",
#     "libyang",
#     "llhttp",
#     "swaylock",
#     "wkhtmltopdf",
#     "js-compute-runtime",
#     "lrzip",
#     "w3m",
#     "ox",
#     "jansson",
#     "nginx",
#     "libgadu",
#     "grep",
#     "libzmq",
#     "mapserver",
#     "screen",
#     "libtorrent",
#     "png-img",
#     "logrotate",
#     "libconfuse",
#     "rtl_433",
#     "atril",
#     "jdk8u",
#     "jq",
#     "libvips",
#     "server",
#     "libde265",
#     "cryptsetup",
#     "lua-nginx-module",
#     "pam-u2f",
#     "mono",
#     "fmt",
#     "oniguruma",
#     "epiphany",
#     "toybox",
#     "aria2",
#     "gifsicle",
#     "libgit2",
#     "patch",
#     "unrar",
#     "transmission",
#     "libguestfs",
#     "wolfMQTT",
#     "rsync",
#     "libmaxminddb",
#     "libgd",
#     "charybdis",
#     "connman",
#     "hiredis",
#     "cpio",
#     "libav",
#     "ipsec",
#     "png2webp",
#     "squashfs-tools",
#     "gcab",
#     "qdecoder",
#     "deark",
#     "tntnet",
#     "abcm2ps",
#     "bind9",
#     "rufus",
#     "electron",
#     "xserver",
#     "domoticz",
#     "barebox",
#     "libtasn1",
#     "olm",
#     "libavif",
#     "mcrouter",
#     "gtk-vnc",
#     "binutils-gdb",
#     "acl",
#     "fluent-bit",
#     "micro-ecc",
#     "nettle",
#     "GameNetworkingSockets",
#     "gnome-online-accounts",
#     "gnome-autoar",
#     "evince",
#     "postsrsd",
#     "mod_md",
#     "gimp",
#     "icoutils",
#     "godot",
#     "kopano-core",
#     "rdesktop",
#     "libgphoto2",
#     "libebml",
#     "libimobiledevice",
#     "winscp",
#     "claws",
#     "chafa",
#     "irssi",
#     "vlc",
#     "core",
#     "varnish-modules",
#     "gthumb",
#     "janet",
#     "folly",
#     "ceph-client",
#     "freeradius-server",
#     "tcpreplay",
#     "edgeless-mariadb",
#     "bpf",
#     "contiki-ng",
#     "mpv",
#     "gnome-settings-daemon",
#     "htmldoc",
#     "bison",
#     "cmark-gfm",
#     "rhonabwy",
#     "src",
#     "bfgminer",
#     "dosfstools",
#     "sqlite",
#     "libetpan",
# }


# --- GitHub API Helpers ---
def get_github_repo_data(repo_full_name: str) -> dict | None:
    """Fetches the main data for a specific repository to check if it's a fork."""

    url: str = f"https://api.github.com/repos/{repo_full_name}"
    try:
        response: requests.Response = requests.get(url, headers=GITHUB_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def get_github_repo_languages(repo_full_name: str) -> dict[str, str]:
    """Fetches the language breakdown for a specific repository."""

    url: str = f"https://api.github.com/repos/{repo_full_name}/languages"
    try:
        # perform HTTP GET request
        response: requests.Response = requests.get(url=url, headers=GITHUB_HEADERS)
        # raise exception in case of error
        response.raise_for_status()
        # encode data in json format (i.e. dict)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching languages for {repo_full_name}: {e}")
        return {}


def find_best_github_match(project_name: str) -> dict | None:
    """Searches for a project and returns the top search result."""

    url: str = (
        f"https://api.github.com/search/repositories?q={project_name}+in:name&sort=stars&order=desc"
    )
    try:
        response: requests.Response = requests.get(url=url, headers=GITHUB_HEADERS)
        response.raise_for_status()
        search_results = response.json()
        if search_results.get("items"):
            return search_results["items"][0]
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error searching for {project_name}: {e}")
        return None


def find_best_gitlab_match(project_name: str) -> dict | None:
    """Searches GitLab for a project and returns the top result."""
    # Note: The default GitLab search host is gitlab.com.

    url = f"https://gitlab.com/api/v4/projects?search={project_name}&order_by=stars_count&sort=desc"
    try:
        response = requests.get(url, headers=GITLAB_HEADERS)
        response.raise_for_status()
        search_results = response.json()
        if search_results:
            return search_results[0]
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error searching GitLab for {project_name}: {e}")
        return None


# --- GitLab API Helpers ---
def get_gitlab_project_data(project_path: str, gitlab_host: str) -> dict | None:
    # GitLab API uses URL-encoded paths (e.g., group/project -> group%2Fproject)

    encoded_path = quote(project_path, safe="")
    url = f"https://{gitlab_host}/api/v4/projects/{encoded_path}"
    try:
        response: requests.Response = requests.get(url, headers=GITLAB_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def get_gitlab_project_languages(project_id: str, gitlab_host: str) -> dict:
    url: str = f"https://{gitlab_host}/api/v4/projects/{project_id}/languages"
    try:
        response: requests.Response = requests.get(url, headers=GITLAB_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {}


def analyze_repositories(fp: str, reference_dataset: str) -> list[str]:
    """Loads a JSONL file, finds the parent repository if it's a fork,
    queries the GitHub API, and returns a list of C++ projects.
    """

    # <---- Check for GitHub token ---->
    if not GITHUB_TOKEN or not GITHUB_TOKEN.startswith("ghp_"):
        raise AttributeError(
            "ERROR: GitHub Personal Access Token is not set.\n"
            "Please create a .env file with your GITHUB_PAT."
        )

    # <---- Load and map verified metadata ---->
    # project_data: list[dict[str,str]] = []
    metadata_repo_map: dict[str, str | None] = {}
    with Loader(desc=f"Loading data from {fp} and buiding mapping "):
        with open(file=fp, mode="r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():  # ignore empty lines
                    continue

                try:
                    entry: dict[str, str] = json.loads(s=line)
                    project_name: str | None = entry.get("project")

                    if project_name and project_name not in metadata_repo_map:
                        repo_url: str | None = entry.get("repo_url")
                        metadata_repo_map[project_name] = repo_url
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line.strip()}")

    # --- identify all unique projects from the main dataset ---
    all_projects: set[str | None] = set()
    with open(
        file=reference_dataset,
        mode="r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            entry: dict[str, str] = json.loads(s=line)
            project_name: str | None = entry.get("project")
            all_projects.add(project_name)

        # --- Step 3: Find projects that need searching ---
    projects_to_search: set[str | None] = all_projects - set(metadata_repo_map.keys())
    print(f"Found {len(metadata_repo_map)} projects with metadata.")
    print(f"Found {len(projects_to_search)} projects requiring a search.")

    # --- search for missing projects ---
    found_repo_map: dict = {}
    if projects_to_search:
        with tqdm(
            iterable=projects_to_search, desc="Analysing repos with no metadata"
        ) as pbar:
            for project_name in pbar:
                if not project_name:
                    continue
                repo_info = find_best_github_match(project_name=project_name)
                if repo_info:
                    found_repo_map[project_name] = repo_info.get("html_url")
                time.sleep(1)

    # <---- combine and analyze repositories ---->
    final_repo_map: dict[str, str | None] = {**metadata_repo_map, **found_repo_map}
    cpp_projects: list[str] = []
    unsupported_repos: list[str] = []

    with tqdm(
        iterable=final_repo_map.items(),
        desc=f"Analyzing {len(final_repo_map)} total unique projects",
    ) as pbar:
        for project_name, repo_url in pbar:
            pbar.set_postfix_str(f"Checking {project_name}")
            time.sleep(1)

            repo_info: dict | None = None
            platform: str | None = None
            path: str = ""

            if repo_url:
                # <---- try to get data from current URL ---->
                parsed_url: ParseResult = urlparse(url=repo_url)
                hostname: str = parsed_url.netloc
                path: str = parsed_url.path.strip("/")
                cleaned_path: str = path[:-4] if path.endswith(".git") else path

                if "github.com" in hostname:
                    repo_info = get_github_repo_data(repo_full_name=cleaned_path)
                    if repo_info:
                        platform = "github"
                elif "gitlab" in hostname:
                    repo_info = get_gitlab_project_data(
                        project_path=cleaned_path, gitlab_host=hostname
                    )
                    if repo_info:
                        platform = "gitlab"

            # <---- if URL failed or was missing, fallback to search ---->
            if not repo_info:
                pbar.set_postfix_str(
                    f"URL failed/missing for '{project_name}'. Searching..."
                )

                # search for best matching project
                gh_match = find_best_github_match(project_name=project_name)
                time.sleep(0.5)
                gl_match = find_best_gitlab_match(project_name=project_name)
                time.sleep(0.5)

                # get start of two best matched projects
                gh_stars: int = gh_match.get("stargazers_count", 0) if gh_match else 0
                gl_stars: int = gl_match.get("star_count", 0) if gl_match else 0

                if gh_match and gh_stars >= gl_stars:
                    repo_info, platform = gh_match, "github"
                elif gl_match:
                    repo_info, platform = gl_match, "gitlab"

            if not repo_info:
                unsupported_repos.append(f"{project_name}: (Not Found)")
                continue

            # <---- Analyze the repository if it was found ---->
            if not repo_info or not platform:
                print(f"\nWarning: Could not find a repository for '{project_name}'.")
                continue

            target_languages: dict[str, str] = {}
            if platform == "github":
                # first, resolve forks to get to the source repo
                if repo_info.get("fork") and repo_info.get("parent"):
                    parent_full_name = repo_info["parent"].get("full_name")
                    pbar.set_postfix_str(
                        f"'{project_name}' is a fork. Checking parent '{parent_full_name}'..."
                    )
                    # Overwrite repo_info with the parent's data
                    repo_info = get_github_repo_data(repo_full_name=parent_full_name)
                    time.sleep(0.5)

                if repo_info and repo_info.get("archived"):
                    pbar.set_postfix_str(
                        f"{repo_info['full_name']} is archived. Searching ..."
                    )
                    canonical_repo: dict | None = find_best_github_match(
                        project_name=project_name
                    )
                    if canonical_repo:
                        target_languages = get_github_repo_languages(
                            canonical_repo["full_name"]
                        )

                elif repo_info:
                    target_languages = get_github_repo_languages(repo_info["full_name"])

            elif platform == "gitlab":
                if repo_info.get("forked_from_project"):
                    parent_id = repo_info["forked_from_project"].get("id")
                    if parent_id:
                        host = urlparse(repo_info.get("web_url", "")).netloc
                        repo_info = get_gitlab_project_data(
                            project_path=str(parent_id), gitlab_host=host
                        )
                        time.sleep(0.5)

                if repo_info and repo_info.get("archived"):
                    pbar.set_postfix_str(
                        f"GitLab project for '{project_name}' is archived. Searching..."
                    )
                    # Fallback to competitive search if archived
                    gh_match = find_best_github_match(project_name=project_name)
                    if gh_match:
                        target_languages = get_github_repo_languages(
                            repo_full_name=gh_match["full_name"]
                        )

                elif repo_info:
                    host = urlparse(repo_info.get("web_url", "")).netloc
                    target_languages = get_gitlab_project_languages(
                        str(repo_info["id"]), host
                    )

            if "C++" in target_languages:
                cpp_projects.append(project_name)

            time.sleep(1)

    # --- save the list of unsupported repos ---
    if unsupported_repos:
        output_file = "./assets/unsupported_repos.txt"
        print(
            f"\nCould not analyze {len(unsupported_repos)} non-GitHub repos. Saving list to {output_file}..."
        )
        with open(file=output_file, mode="w", encoding="utf-8") as f:
            for line in unsupported_repos:
                f.write(line + "\n")

    return cpp_projects
