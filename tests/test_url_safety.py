import asyncio
import socket
import ssl

import anyio
import pytest
from anyio.streams.tls import TLSStream

from app.config import Settings
from app.services.url_safety import (
    UnsafeUrlError,
    WebsiteDnsError,
    parse_website_url,
    resolve_destination,
)
from app.services.website_extractor import WebsiteExtractor

pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/products",
        "http://example.com:8080/path",
        "http://8.8.8.8/",
        "https://[2001:4860::8888]/",
        "https://example.com/products%20and%20services",
    ],
)
async def test_public_http_urls(url):
    assert str(parse_website_url(url)) == url


@pytest.mark.parametrize(
    "url",
    [
        "example.com",
        "http:///missing-host",
        "https://invalid host.test",
        "https://example.com:65536",
        "ftp://example.com",
        "file:///tmp/site.html",
        "https://localhost",
        "https://app.localhost.",
        "https://localhost.localdomain",
        "http://192.168.0.1",
        "http://[fd00::21]",
        "http://127.0.0.1",
        "http://[::1]",
        "http://169.254.169.254",
        "http://[fe80::1]",
        "http://[::ffff:127.0.0.1]",
        "http://100.64.0.1",
        "http://224.0.0.1",
        "http://192.0.2.1",
        "http://192.88.99.2",
        "http://192.31.196.1",
        "http://192.52.193.1",
        "http://192.175.48.1",
        "http://[2620:4f:8000::1]",
        "http://0.0.0.0",
        "http://10.0.0.1",
        "http://172.16.0.1",
        "http://192.0.0.9",
        "http://198.18.0.1",
        "http://198.51.100.1",
        "http://203.0.113.1",
        "http://240.0.0.1",
        "http://255.255.255.255",
        "http://169.254.170.2",
        "http://168.63.129.16",
        "http://100.100.100.200",
        "http://[fd00:ec2::254]",
        "http://[::]",
        "http://[::ffff:8.8.8.8]",
        "http://[ff02::1]",
        "http://[100::1]",
        "http://[100:0:0:1::1]",
        "http://[64:ff9b::1]",
        "http://[64:ff9b:1::1]",
        "http://[2001:db8::1]",
        "http://[2001::1]",
        "http://[2002::1]",
        "http://[2000::1]",
        "http://[2d00::1]",
        "http://[3ffe::1]",
        "http://[3fff::1]",
        "http://[5f00::1]",
        "http://[fec0::1]",
        "http://[fe80::1%25en0]",
        "http://127.1/",
        "http://0177.0.0.1/",
        "http://0x7f.0.0.1/",
        "http://2130706433/",
        "http://%31%32%37.0.0.1/",
        "https://user:password@example.com",
        "https://@example.com",
    ],
)
async def test_rejects_unsafe_urls(url):
    with pytest.raises(UnsafeUrlError):
        parse_website_url(url)


@pytest.mark.parametrize("address", ["8.8.8.8", "2001:4860::8888"])
async def test_public_hostname_is_pinned(monkeypatch, address):
    lookup_hosts = []

    async def getaddrinfo(host, port, **kwargs):
        lookup_hosts.append(host)
        return [(0, 0, 0, "", (address, port))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)

    destination = await resolve_destination(parse_website_url("https://example.com:8443/services"))

    assert lookup_hosts == ["example.com."]
    assert destination.connection_url.host == address
    assert destination.connection_url.port == 8443
    assert destination.connection_url.path == "/services"
    assert destination.tls_hostname == "example.com"


@pytest.mark.parametrize(
    "addresses",
    [
        ["10.10.0.1"],
        ["8.8.8.8", "fd00::8"],
        ["169.254.169.254", "2001:4860::8888"],
        ["2001:4860::8888", "168.63.129.16"],
        ["8.8.8.8", "::ffff:8.8.8.8"],
        ["8.8.8.8", "192.31.196.1"],
    ],
)
async def test_rejects_any_private_dns_result(monkeypatch, addresses):
    async def getaddrinfo(host, port, **kwargs):
        return [(0, 0, 0, "", (address, port)) for address in addresses]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)
    with pytest.raises(UnsafeUrlError):
        await resolve_destination(parse_website_url("https://example.com"))


async def test_dns_failure(monkeypatch):
    async def getaddrinfo(*args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "No address")

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)
    with pytest.raises(WebsiteDnsError, match="could not be resolved"):
        await resolve_destination(parse_website_url("https://example.com"))


async def test_httpcore_connects_to_pinned_address_with_original_tls_hostname(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://unused-proxy.example:3128")
    monkeypatch.setenv("NO_PROXY", "")
    lookups = []
    connections = []
    tls_hosts = []
    sent = []
    replies = [
        b"HTTP/1.1 302 Found\r\nLocation: https://second.example.com:8443/final\r\n"
        b"Content-Length: 0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 15\r\n\r\n"
        b"<p>Services</p>",
    ]

    class FakeStream:
        def __init__(self, reply):
            self.reply = reply

        def extra(self, attribute, default=None):
            return default

        async def receive(self, max_bytes):
            data, self.reply = self.reply[:max_bytes], self.reply[max_bytes:]
            return data

        async def send(self, item):
            sent.append(item)

        async def aclose(self):
            pass

    async def getaddrinfo(host, port, **kwargs):
        lookups.append(host)
        # The validated address is shared by both domains; an unexpected resolution fails.
        assert host in {"example.com.", "second.example.com."}
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", port))]

    async def connect_tcp(*, remote_host, remote_port, **kwargs):
        connections.append((remote_host, remote_port))
        return FakeStream(replies.pop(0))

    async def wrap(stream, *, hostname, ssl_context, **kwargs):
        assert ssl_context.check_hostname and ssl_context.verify_mode == ssl.CERT_REQUIRED
        tls_hosts.append(hostname)
        return stream

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(anyio, "connect_tcp", connect_tcp)
    monkeypatch.setattr(TLSStream, "wrap", wrap)

    result = await WebsiteExtractor(Settings()).extract("https://example.com:8443/path")

    assert result.success
    assert lookups == ["example.com.", "second.example.com."]
    assert connections == [("8.8.8.8", 8443), ("8.8.8.8", 8443)]
    assert tls_hosts == ["example.com", "second.example.com"]
    assert b"Host: example.com:8443\r\n" in sent[0]
    assert b"Host: second.example.com:8443\r\n" in sent[1]
