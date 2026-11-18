import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx


class UnsafeUrlError(ValueError):
    """The URL is malformed or points to a prohibited destination."""


class WebsiteDnsError(ConnectionError):
    """The hostname could not be resolved."""


IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
_DOMAIN_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.I | re.ASCII)
_NUMERIC_LABEL = re.compile(r"(?:[0-9]+|0x[0-9a-f]+)", re.I | re.ASCII)
# Also deny globally reachable IANA special-purpose assignments and the
# publicly numbered Azure platform endpoint, neither of which is a website target.
_IPV4_SPECIAL = tuple(
    ipaddress.IPv4Network(network)
    for network in (
        "192.0.0.0/24", "192.31.196.0/24", "192.52.193.0/24",
        "192.88.99.0/24", "192.175.48.0/24", "168.63.129.16/32",
    )
)
# IANA IPv6 Global Unicast Address Space, 2025-10-10. Allow RIR allocations
# only, so unallocated space inside 2000::/3 also fails closed.
_IPV6_PUBLIC = tuple(
    ipaddress.IPv6Network(network)
    for network in (
        "2001:200::/23", "2001:400::/23", "2001:600::/23", "2001:800::/22",
        "2001:c00::/23", "2001:e00::/23", "2001:1200::/23", "2001:1400::/22",
        "2001:1800::/23", "2001:1a00::/23", "2001:1c00::/22", "2001:2000::/19",
        "2001:4000::/23", "2001:4200::/23", "2001:4400::/23", "2001:4600::/23",
        "2001:4800::/23", "2001:4a00::/23", "2001:4c00::/23", "2001:5000::/20",
        "2001:8000::/19", "2001:a000::/20", "2001:b000::/20", "2003::/18",
        "2400::/12", "2410::/12", "2600::/12", "2610::/23", "2620::/23", "2630::/12",
        "2800::/12", "2a00::/12", "2a10::/12", "2c00::/12",
    )
)
_IPV6_AS112 = ipaddress.IPv6Network("2620:4f:8000::/48")
_LOCALHOST_ALIASES = {"localhost.localdomain", "localhost4.localdomain4", "localhost6.localdomain6"}


@dataclass(frozen=True)
class ValidatedDestination:
    url: httpx.URL
    address: IPAddress

    @property
    def connection_url(self) -> httpx.URL:
        return self.url.copy_with(host=str(self.address))

    @property
    def tls_hostname(self) -> str:
        return self.url.raw_host.decode("ascii").rstrip(".")


def _check_public_address(address: IPAddress) -> None:
    if isinstance(address, ipaddress.IPv6Address):
        # Requiring native global unicast also excludes IPv4-mapped addresses.
        if address in _IPV6_AS112 or not any(
            address in network for network in _IPV6_PUBLIC
        ):
            raise UnsafeUrlError("The website must use a publicly routable IP address.")
    elif any(address in network for network in _IPV4_SPECIAL):
        raise UnsafeUrlError("The website must use a publicly routable IP address.")

    if not address.is_global or address.is_multicast or address.is_reserved:
        raise UnsafeUrlError("The website must use a publicly routable IP address.")


def parse_website_url(value: str) -> httpx.URL:
    """Check URL syntax and literal destinations without performing DNS resolution."""
    if "\\" in value or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise UnsafeUrlError("Provide a valid absolute HTTP or HTTPS website URL.")

    try:
        authority = urlsplit(value).netloc
        url = httpx.URL(value)
    except (ValueError, httpx.InvalidURL) as exc:
        raise UnsafeUrlError("Provide a valid absolute HTTP or HTTPS website URL.") from exc

    if url.scheme not in {"http", "https"} or not authority or not url.raw_host:
        raise UnsafeUrlError("Provide a valid absolute HTTP or HTTPS website URL.")
    if "@" in authority:
        raise UnsafeUrlError("Website URLs cannot contain credentials.")
    if "%" in authority or authority.endswith(":") or (
        url.port is not None and not 1 <= url.port <= 65535
    ):
        raise UnsafeUrlError("The website hostname or port is invalid.")

    raw_host = url.raw_host.decode("ascii")
    try:
        address = ipaddress.ip_address(raw_host)
    except ValueError:
        hostname = raw_host.removesuffix(".")
        labels = hostname.split(".")
        if (
            authority.startswith("[")
            or len(labels) < 2
            or len(hostname) > 253
            or not all(_DOMAIN_LABEL.fullmatch(label) for label in labels)
            or hostname in _LOCALHOST_ALIASES
            or hostname.endswith((".localhost", ".local"))
            or all(_NUMERIC_LABEL.fullmatch(label) for label in labels)
        ):
            raise UnsafeUrlError("Provide a valid public website hostname.")
    else:
        _check_public_address(address)

    return url.copy_with(fragment=None)


async def resolve_destination(url: httpx.URL) -> ValidatedDestination:
    """Resolve a parsed URL and pin a connection to one of its checked addresses."""
    hostname = url.raw_host.decode("ascii").rstrip(".")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        port = url.port or (443 if url.scheme == "https" else 80)
        try:
            records = await asyncio.get_running_loop().getaddrinfo(
                hostname + ".", port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
            )
        except OSError as exc:
            raise WebsiteDnsError("The website hostname could not be resolved.") from exc
        addresses = [ipaddress.ip_address(record[4][0]) for record in records]
        if not addresses:
            raise WebsiteDnsError("The website hostname could not be resolved.")
        for resolved in addresses:
            _check_public_address(resolved)
        address = addresses[0]

    _check_public_address(address)
    return ValidatedDestination(url=url, address=address)
