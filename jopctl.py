#!/usr/bin/env python3
"""
Joplin CLI helper.

This tool wraps the Joplin Data API (clipper server) to inspect and manage
notebooks, notes, tags, and resources from the terminal.
"""

import json
import os
import sys
from functools import wraps
from pathlib import Path
from typing import IO, Any, Dict, Iterable, List, Optional

import click

try:
    import requests
except ImportError as exc:  # pragma: no cover - dependency guard
    print("requests is required. Install with: pip install requests", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # YAML output is optional


DEFAULT_PORT = os.environ.get("JOPLIN_API_PORT", "41184")
DEFAULT_TOKEN = os.environ.get("JOPLIN_API_TOKEN")
OUTPUT_FORMATS = ["text", "json", "yaml"]


def cli_error(message: str) -> None:
    """Raise a Click-friendly error."""
    raise click.ClickException(message)


class JoplinClient:
    """Minimal Joplin Data API client."""

    def __init__(self, port: str, token: str):
        self.base_url = f"http://localhost:{port}"
        self.token = token

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if params is None:
            params = {}
        params["token"] = self.token
        resp = requests.request(
            method,
            f"{self.base_url}{path}",
            params=params,
            json=json_body,
            data=data,
            files=files,
            timeout=30,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if resp.content:
            if (
                "application/json" in content_type
                or resp.text.startswith("{")
                or resp.text.startswith("[")
            ):
                return resp.json()
            return resp.text
        return {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params)

    def _collect_paginated(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        page_params = dict(params or {})
        results: List[Dict[str, Any]] = []
        page = 1
        while True:
            page_params.update({"page": page})
            data = self._get(path, page_params)
            if isinstance(data, dict) and "items" in data:
                results.extend(data["items"])
                if data.get("has_more"):
                    page += 1
                    continue
            else:
                # non-paginated response
                results = data if isinstance(data, list) else [data]
            break
        return results

    def _post(
        self,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self._request("POST", path, params=params, json_body=json_body, data=data, files=files)

    def _put(
        self,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self._request("PUT", path, params=params, json_body=json_body, data=data, files=files)

    def _delete(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("DELETE", path, params=params)

    def ping(self) -> bool:
        try:
            res = self._get("/ping", {})
            return isinstance(res, str) and res.strip() == "JoplinClipperServer"
        except Exception:
            return False

    def folders(self) -> List[Dict[str, Any]]:
        return self._collect_paginated("/folders")

    def folder(self, folder_id: str) -> Dict[str, Any]:
        return self._get(f"/folders/{folder_id}")

    def notes(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"order_by": "updated_time", "order_dir": "DESC"}
        if limit:
            params["limit"] = limit
        return self._collect_paginated("/notes", params)

    def folder_notes(self, folder_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        return self._collect_paginated(f"/folders/{folder_id}/notes", params)

    def note(self, note_id: str, with_body: bool = False) -> Dict[str, Any]:
        fields = "id,title,updated_time,created_time,parent_id"
        if with_body:
            fields += ",body"
        return self._get(f"/notes/{note_id}", {"fields": fields})

    def search(self, query: str, item_type: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"query": query}
        if item_type:
            params["type"] = item_type
        if limit:
            params["limit"] = limit
        data = self._get("/search", params)
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        return data if isinstance(data, list) else []

    def tags(self) -> List[Dict[str, Any]]:
        return self._collect_paginated("/tags")

    def tag_notes(self, tag_id: str) -> List[Dict[str, Any]]:
        return self._collect_paginated(f"/tags/{tag_id}/notes")

    def resources(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        return self._collect_paginated("/resources", params)

    def resource(self, resource_id: str) -> Dict[str, Any]:
        return self._get(f"/resources/{resource_id}")

    def create_folder(self, title: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"title": title}
        if parent_id:
            payload["parent_id"] = parent_id
        return self._post("/folders", json_body=payload)

    def update_folder(
        self, folder_id: str, title: Optional[str] = None, parent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if parent_id is not None:
            payload["parent_id"] = parent_id
        return self._put(f"/folders/{folder_id}", json_body=payload)

    def delete_folder(self, folder_id: str) -> Any:
        return self._delete(f"/folders/{folder_id}")

    def create_note(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/notes", json_body=payload)

    def update_note(self, note_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._put(f"/notes/{note_id}", json_body=payload)

    def delete_note(self, note_id: str, permanent: bool = False) -> Any:
        params = {"permanent": int(permanent)} if permanent else None
        return self._delete(f"/notes/{note_id}", params=params)

    def create_tag(self, title: str) -> Dict[str, Any]:
        return self._post("/tags", json_body={"title": title})

    def update_tag(self, tag_id: str, title: str) -> Dict[str, Any]:
        return self._put(f"/tags/{tag_id}", json_body={"title": title})

    def delete_tag(self, tag_id: str) -> Any:
        return self._delete(f"/tags/{tag_id}")

    def tag_note(self, tag_id: str, note_id: str) -> Any:
        return self._post(f"/tags/{tag_id}/notes", json_body={"id": note_id})

    def untag_note(self, tag_id: str, note_id: str) -> Any:
        return self._delete(f"/tags/{tag_id}/notes/{note_id}")

    def create_resource(
        self,
        filename: str,
        file_obj: IO[bytes],
        props: Optional[Dict[str, Any]] = None,
        mime: Optional[str] = None,
    ) -> Dict[str, Any]:
        files: Dict[str, Any] = {
            "data": (filename, file_obj, mime or "application/octet-stream"),
        }
        if props:
            files["props"] = (None, json.dumps(props))
        return self._post("/resources", files=files)

    def update_resource(
        self,
        resource_id: str,
        filename: Optional[str] = None,
        file_obj: Optional[IO[bytes]] = None,
        props: Optional[Dict[str, Any]] = None,
        mime: Optional[str] = None,
    ) -> Dict[str, Any]:
        if file_obj:
            files: Dict[str, Any] = {
                "data": (filename or resource_id, file_obj, mime or "application/octet-stream"),
            }
            if props:
                files["props"] = (None, json.dumps(props))
            return self._put(f"/resources/{resource_id}", files=files)
        payload = props or {}
        return self._put(f"/resources/{resource_id}", json_body=payload or None)

    def delete_resource(self, resource_id: str) -> Any:
        return self._delete(f"/resources/{resource_id}")


def format_table(headers: List[str], rows: Iterable[List[Any]]) -> str:
    """Render a simple ASCII table."""
    col_widths = [len(h) for h in headers]
    str_rows: List[List[str]] = []
    for row in rows:
        str_row = ["" if v is None else str(v) for v in row]
        str_rows.append(str_row)
        for idx, val in enumerate(str_row):
            col_widths[idx] = max(col_widths[idx], len(val))
    fmt = "  ".join(f"{{:{w}}}" for w in col_widths)
    lines = [fmt.format(*headers)]
    lines.append("  ".join("-" * w for w in col_widths))
    lines.extend(fmt.format(*r) for r in str_rows)
    return "\n".join(lines)


def print_output(data: Any, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    if fmt == "yaml":
        if yaml is None:
            print("YAML output requires pyyaml. Falling back to JSON.", file=sys.stderr)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(yaml.safe_dump(data, sort_keys=False))
        return
    # text
    if isinstance(data, str):
        print(data)
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


def resolve_folder_id(identifier: str, folders: List[Dict[str, Any]]) -> Optional[str]:
    """Resolve folder by ID or exact title (case-insensitive)."""
    for f in folders:
        if f.get("id") == identifier:
            return identifier
    matches = [f for f in folders if f.get("title", "").lower() == identifier.lower()]
    if len(matches) == 1:
        return matches[0].get("id")
    if len(matches) > 1:
        root_matches = [f for f in matches if not f.get("parent_id")]
        if len(root_matches) == 1:
            return root_matches[0].get("id")
    return None


def require_folder_id(
    identifier: str, client: "JoplinClient", folders: Optional[List[Dict[str, Any]]] = None
) -> str:
    known = folders or client.folders()
    folder_id = resolve_folder_id(identifier, known)
    if not folder_id:
        cli_error(f"No notebook found for '{identifier}'. Use ID or exact title.")
    return folder_id


def read_text_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        cli_error(f"Unable to read file '{path}': {exc}")


def parse_json_arg(arg_name: str, value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        cli_error(f"Invalid JSON for {arg_name}: {exc}")
    if not isinstance(parsed, dict):
        cli_error(f"{arg_name} expects a JSON object.")
    return parsed


def resolve_body_argument(text_value: Optional[str], file_path: Optional[str], flag: str) -> Optional[str]:
    if text_value and file_path:
        cli_error(f"Specify {flag} text or file, not both.")
    if file_path:
        return read_text_file(file_path)
    return text_value


def build_folder_tree(folders: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return dict of folder_id -> folder (with children) built from flat list."""
    nodes: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for folder in folders:
        folder_id = folder.get("id")
        if not folder_id:
            continue
        node = dict(folder)
        node["children"] = []
        nodes[folder_id] = node
        order.append(folder_id)

    for folder_id in order:
        node = nodes[folder_id]
        parent_id = node.get("parent_id")
        if parent_id and parent_id in nodes:
            nodes[parent_id]["children"].append(node)

    return nodes


def with_client(handler):
    """Click helper to inject the client and handle HTTP errors."""

    @wraps(handler)
    @click.pass_context
    def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> Any:
        client: JoplinClient = ctx.obj["client"]
        local_format = kwargs.pop("local_format", None)
        output_format: str = local_format or ctx.obj["format"]
        try:
            return handler(client, output_format, *args, **kwargs)
        except requests.HTTPError as exc:
            raise click.ClickException(
                f"HTTP error: {exc.response.status_code} {exc.response.text}"
            ) from exc
        except requests.RequestException as exc:
            raise click.ClickException(f"Request error: {exc}") from exc

    return wrapper


def format_option(func):
    """Allow per-command --format overrides placed after the subcommand."""

    return click.option(
        "--format",
        "local_format",
        type=click.Choice(OUTPUT_FORMATS),
        help="Override output format for this command.",
    )(func)


def handle_auth(client: JoplinClient, _output_format: Optional[str] = None) -> None:
    ok = client.ping()
    if ok:
        click.echo(f"Connected to {client.base_url} ✔")
        return
    cli_error(f"Unable to reach {client.base_url}. Is the clipper server running?")


def handle_notebook_list(client: JoplinClient, output_format: str) -> None:
    folders = client.folders()
    rows = [[f.get("id"), f.get("title", ""), f.get("parent_id", "")] for f in folders]
    if output_format == "text":
        print(format_table(["id", "title", "parent"], rows))
    else:
        print_output([{"id": r[0], "title": r[1], "parent_id": r[2]} for r in rows], output_format)


def handle_notebook_show(client: JoplinClient, output_format: str, identifier: str) -> None:
    folders = client.folders()
    folder_id = resolve_folder_id(identifier, folders)
    if not folder_id:
        cli_error(f"No notebook found for '{identifier}'. Use ID or exact title.")
    folder_tree = build_folder_tree(folders)
    folder = folder_tree.get(folder_id)
    if not folder:
        folder = client.folder(folder_id)
    print_output(folder, output_format)


def handle_notebook_create(
    client: JoplinClient, output_format: str, title: str, parent: Optional[str]
) -> None:
    parent_id = None
    folders: Optional[List[Dict[str, Any]]] = None
    if parent:
        folders = client.folders()
        parent_id = require_folder_id(parent, client, folders)
    notebook = client.create_folder(title, parent_id=parent_id)
    print_output(notebook, output_format)


def handle_notebook_update(
    client: JoplinClient,
    output_format: str,
    identifier: str,
    title: Optional[str],
    parent: Optional[str],
) -> None:
    if title is None and parent is None:
        cli_error("Provide --title and/or --parent to update the notebook.")
    folders = client.folders()
    folder_id = resolve_folder_id(identifier, folders)
    if not folder_id:
        cli_error(f"No notebook found for '{identifier}'. Use ID or exact title.")
    parent_id = None
    if parent:
        parent_id = require_folder_id(parent, client, folders)
    notebook = client.update_folder(folder_id, title=title, parent_id=parent_id)
    print_output(notebook, output_format)


def handle_notebook_delete(client: JoplinClient, output_format: str, identifier: str) -> None:
    folders = client.folders()
    folder_id = resolve_folder_id(identifier, folders)
    if not folder_id:
        cli_error(f"No notebook found for '{identifier}'. Use ID or exact title.")
    result = client.delete_folder(folder_id)
    print_output(result or {"deleted": folder_id}, output_format)


def handle_note_list(
    client: JoplinClient, output_format: str, notebook: Optional[str], limit: Optional[int]
) -> None:
    if notebook:
        folders = client.folders()
        folder_id = resolve_folder_id(notebook, folders)
        if not folder_id:
            cli_error(f"No notebook found for '{notebook}'.")
        notes = client.folder_notes(folder_id, limit)
    else:
        notes = client.notes(limit)
    rows = [
        [
            note.get("id"),
            note.get("title", ""),
            note.get("updated_time"),
            note.get("parent_id", ""),
        ]
        for note in notes
    ]
    if output_format == "text":
        print(format_table(["id", "title", "updated", "notebook"], rows))
    else:
        print_output(
            [
                {"id": r[0], "title": r[1], "updated_time": r[2], "parent_id": r[3]}
                for r in rows
            ],
            output_format,
        )


def handle_note_show(client: JoplinClient, output_format: str, note_id: str, include_body: bool) -> None:
    note = client.note(note_id, with_body=include_body)
    print_output(note, output_format)


def handle_note_create(
    client: JoplinClient,
    output_format: str,
    title: str,
    body_text: Optional[str],
    body_file: Optional[str],
    body_html: Optional[str],
    notebook: Optional[str],
    data: Optional[str],
) -> None:
    payload: Dict[str, Any] = {"title": title}
    body = resolve_body_argument(body_text, body_file, "--body")
    if body is not None:
        payload["body"] = body
    if body_html:
        payload["body_html"] = body_html
    if notebook:
        payload["parent_id"] = require_folder_id(notebook, client)
    extra = parse_json_arg("--data", data)
    payload.update(extra)
    note = client.create_note(payload)
    print_output(note, output_format)


def handle_note_update(
    client: JoplinClient,
    output_format: str,
    note_id: str,
    title: Optional[str],
    body_text: Optional[str],
    body_file: Optional[str],
    body_html: Optional[str],
    notebook: Optional[str],
    data: Optional[str],
) -> None:
    payload: Dict[str, Any] = {}
    if title is not None:
        payload["title"] = title
    body = resolve_body_argument(body_text, body_file, "--body")
    if body is not None:
        payload["body"] = body
    if body_html is not None:
        payload["body_html"] = body_html
    if notebook:
        payload["parent_id"] = require_folder_id(notebook, client)
    payload.update(parse_json_arg("--data", data))
    if not payload:
        cli_error("Provide at least one field to update the note.")
    note = client.update_note(note_id, payload)
    print_output(note, output_format)


def handle_note_delete(client: JoplinClient, output_format: str, note_id: str, permanent: bool) -> None:
    result = client.delete_note(note_id, permanent=permanent)
    print_output(result or {"deleted": note_id}, output_format)


def handle_search(
    client: JoplinClient, output_format: str, query: str, item_type: Optional[str], limit: Optional[int]
) -> None:
    results = client.search(query, item_type, limit)
    if output_format == "text":
        rows = [[item.get("id"), item.get("title", ""), item.get("type_", "")] for item in results]
        print(format_table(["id", "title", "type"], rows))
    else:
        print_output(results, output_format)


def handle_tag_list(client: JoplinClient, output_format: str) -> None:
    tags = client.tags()
    rows = [[t.get("id"), t.get("title", "")] for t in tags]
    if output_format == "text":
        print(format_table(["id", "title"], rows))
    else:
        print_output([{"id": r[0], "title": r[1]} for r in rows], output_format)


def handle_tag_notes(client: JoplinClient, output_format: str, tag_id: str) -> None:
    notes = client.tag_notes(tag_id)
    rows = [[n.get("id"), n.get("title", ""), n.get("updated_time")] for n in notes]
    if output_format == "text":
        print(format_table(["id", "title", "updated"], rows))
    else:
        print_output([{"id": r[0], "title": r[1], "updated_time": r[2]} for r in rows], output_format)


def handle_tag_create(client: JoplinClient, output_format: str, title: str) -> None:
    tag = client.create_tag(title)
    print_output(tag, output_format)


def handle_tag_update(client: JoplinClient, output_format: str, tag_id: str, title: str) -> None:
    tag = client.update_tag(tag_id, title)
    print_output(tag, output_format)


def handle_tag_delete(client: JoplinClient, output_format: str, tag_id: str) -> None:
    result = client.delete_tag(tag_id)
    print_output(result or {"deleted": tag_id}, output_format)


def handle_tag_attach(client: JoplinClient, output_format: str, tag_id: str, note_id: str) -> None:
    result = client.tag_note(tag_id, note_id)
    print_output(result or {"tagged": note_id}, output_format)


def handle_tag_detach(client: JoplinClient, output_format: str, tag_id: str, note_id: str) -> None:
    result = client.untag_note(tag_id, note_id)
    print_output(result or {"untagged": note_id}, output_format)


def handle_resource_list(client: JoplinClient, output_format: str, limit: Optional[int]) -> None:
    resources = client.resources(limit)
    rows = [[r.get("id"), r.get("title", ""), r.get("mime", "")] for r in resources]
    if output_format == "text":
        print(format_table(["id", "title", "mime"], rows))
    else:
        print_output([{"id": r[0], "title": r[1], "mime": r[2]} for r in rows], output_format)


def handle_resource_show(client: JoplinClient, output_format: str, resource_id: str) -> None:
    resource = client.resource(resource_id)
    print_output(resource, output_format)


def build_resource_props(data: Optional[str], title: Optional[str], mime: Optional[str]) -> Dict[str, Any]:
    props = parse_json_arg("--data", data)
    if title is not None:
        props["title"] = title
    if mime is not None:
        props["mime"] = mime
    return props


def handle_resource_create(
    client: JoplinClient,
    output_format: str,
    file_path_str: str,
    title: Optional[str],
    mime: Optional[str],
    data: Optional[str],
) -> None:
    file_path = Path(file_path_str)
    if not file_path.is_file():
        cli_error(f"Resource file '{file_path_str}' not found.")
    props = build_resource_props(data, title, mime)
    with file_path.open("rb") as fh:
        resource = client.create_resource(file_path.name, fh, props or None, mime=mime)
    print_output(resource, output_format)


def handle_resource_update(
    client: JoplinClient,
    output_format: str,
    resource_id: str,
    file_path_str: Optional[str],
    title: Optional[str],
    mime: Optional[str],
    data: Optional[str],
) -> None:
    props = build_resource_props(data, title, mime)
    if not props and not file_path_str:
        cli_error("Provide --file or metadata fields to update the resource.")
    resource: Dict[str, Any]
    if file_path_str:
        file_path = Path(file_path_str)
        if not file_path.is_file():
            cli_error(f"Resource file '{file_path_str}' not found.")
        with file_path.open("rb") as fh:
            resource = client.update_resource(
                resource_id,
                filename=file_path.name,
                file_obj=fh,
                props=props or None,
                mime=mime,
            )
    else:
        resource = client.update_resource(resource_id, props=props)
    print_output(resource, output_format)


def handle_resource_delete(client: JoplinClient, output_format: str, resource_id: str) -> None:
    result = client.delete_resource(resource_id)
    print_output(result or {"deleted": resource_id}, output_format)


@click.group()
@click.option("--port", default=DEFAULT_PORT, show_default=True, help="Joplin API port.")
@click.option(
    "--token",
    default=DEFAULT_TOKEN,
    envvar="JOPLIN_API_TOKEN",
    help="Joplin API token (required).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(OUTPUT_FORMATS),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.pass_context
def cli(ctx: click.Context, port: str, token: Optional[str], output_format: str) -> None:
    """CLI for the Joplin Data API."""

    if not token and not ctx.resilient_parsing:
        raise click.UsageError("API token is required. Set JOPLIN_API_TOKEN or pass --token.")
    ctx.ensure_object(dict)
    ctx.obj["client"] = JoplinClient(port, token)
    ctx.obj["format"] = output_format


@cli.group()
def auth() -> None:
    """Connection utilities."""


@auth.command("status")
@format_option
@with_client
def auth_status(client: JoplinClient, output_format: str) -> None:  # pragma: no cover - CLI wiring
    handle_auth(client, output_format)


@cli.group()
def notebook() -> None:
    """Notebook commands."""


@notebook.command("list")
@format_option
@with_client
def notebook_list(client: JoplinClient, output_format: str) -> None:  # pragma: no cover - CLI wiring
    handle_notebook_list(client, output_format)


@notebook.command("show")
@click.argument("identifier")
@format_option
@with_client
def notebook_show(client: JoplinClient, output_format: str, identifier: str) -> None:  # pragma: no cover
    handle_notebook_show(client, output_format, identifier)


@notebook.command("create")
@click.argument("title")
@click.option("--parent", help="Parent notebook ID or exact title.")
@format_option
@with_client
def notebook_create(
    client: JoplinClient, output_format: str, title: str, parent: Optional[str]
) -> None:  # pragma: no cover - CLI wiring
    handle_notebook_create(client, output_format, title, parent)


@notebook.command("update")
@click.argument("identifier")
@click.option("--title", help="New title.")
@click.option("--parent", help="Parent notebook ID or exact title.")
@format_option
@with_client
def notebook_update(
    client: JoplinClient,
    output_format: str,
    identifier: str,
    title: Optional[str],
    parent: Optional[str],
) -> None:  # pragma: no cover - CLI wiring
    handle_notebook_update(client, output_format, identifier, title, parent)


@notebook.command("delete")
@click.argument("identifier")
@format_option
@with_client
def notebook_delete(client: JoplinClient, output_format: str, identifier: str) -> None:  # pragma: no cover
    handle_notebook_delete(client, output_format, identifier)


@cli.group()
def note() -> None:
    """Note commands."""


@note.command("list")
@click.option("--notebook", help="Notebook ID or exact title to filter.")
@click.option("--limit", type=int, help="Limit number of notes.")
@format_option
@with_client
def note_list(
    client: JoplinClient, output_format: str, notebook: Optional[str], limit: Optional[int]
) -> None:  # pragma: no cover - CLI wiring
    handle_note_list(client, output_format, notebook, limit)


@note.command("show")
@click.argument("note_id")
@click.option("--body", "include_body", is_flag=True, help="Include body content.")
@format_option
@with_client
def note_show(
    client: JoplinClient, output_format: str, note_id: str, include_body: bool
) -> None:  # pragma: no cover - CLI wiring
    handle_note_show(client, output_format, note_id, include_body)


@note.command("create")
@click.option("--title", required=True, help="Note title.")
@click.option("--body", "body_text", help="Markdown body text.")
@click.option("--body-file", help="Path to Markdown body text.")
@click.option("--body-html", help="HTML body text.")
@click.option("--notebook", help="Notebook ID or exact title for the note.")
@click.option("--data", help="Additional JSON object merged into the payload.")
@format_option
@with_client
def note_create(
    client: JoplinClient,
    output_format: str,
    title: str,
    body_text: Optional[str],
    body_file: Optional[str],
    body_html: Optional[str],
    notebook: Optional[str],
    data: Optional[str],
) -> None:  # pragma: no cover - CLI wiring
    handle_note_create(client, output_format, title, body_text, body_file, body_html, notebook, data)


@note.command("update")
@click.argument("note_id")
@click.option("--title", help="Note title.")
@click.option("--body", "body_text", help="Markdown body text.")
@click.option("--body-file", help="Path to Markdown body text.")
@click.option("--body-html", help="HTML body text.")
@click.option("--notebook", help="Notebook ID or exact title for the note.")
@click.option("--data", help="Additional JSON object merged into the payload.")
@format_option
@with_client
def note_update(
    client: JoplinClient,
    output_format: str,
    note_id: str,
    title: Optional[str],
    body_text: Optional[str],
    body_file: Optional[str],
    body_html: Optional[str],
    notebook: Optional[str],
    data: Optional[str],
) -> None:  # pragma: no cover - CLI wiring
    handle_note_update(
        client, output_format, note_id, title, body_text, body_file, body_html, notebook, data
    )


@note.command("delete")
@click.argument("note_id")
@click.option("--permanent", is_flag=True, help="Permanently delete the note.")
@format_option
@with_client
def note_delete(
    client: JoplinClient, output_format: str, note_id: str, permanent: bool
) -> None:  # pragma: no cover - CLI wiring
    handle_note_delete(client, output_format, note_id, permanent)


@cli.command()
@click.argument("query")
@click.option("--type", "item_type", type=click.Choice(["note", "folder", "tag", "resource"]), help="Item type filter.")
@click.option("--limit", type=int, help="Limit number of results.")
@format_option
@with_client
def search(
    client: JoplinClient, output_format: str, query: str, item_type: Optional[str], limit: Optional[int]
) -> None:  # pragma: no cover - CLI wiring
    handle_search(client, output_format, query, item_type, limit)


@cli.group()
def tag() -> None:
    """Tag commands."""


@tag.command("list")
@format_option
@with_client
def tag_list(client: JoplinClient, output_format: str) -> None:  # pragma: no cover - CLI wiring
    handle_tag_list(client, output_format)


@tag.command("notes")
@click.argument("tag_id")
@format_option
@with_client
def tag_notes(client: JoplinClient, output_format: str, tag_id: str) -> None:  # pragma: no cover
    handle_tag_notes(client, output_format, tag_id)


@tag.command("create")
@click.argument("title")
@format_option
@with_client
def tag_create(client: JoplinClient, output_format: str, title: str) -> None:  # pragma: no cover - CLI wiring
    handle_tag_create(client, output_format, title)


@tag.command("update")
@click.argument("tag_id")
@click.argument("title")
@format_option
@with_client
def tag_update(
    client: JoplinClient, output_format: str, tag_id: str, title: str
) -> None:  # pragma: no cover - CLI wiring
    handle_tag_update(client, output_format, tag_id, title)


@tag.command("delete")
@click.argument("tag_id")
@format_option
@with_client
def tag_delete(client: JoplinClient, output_format: str, tag_id: str) -> None:  # pragma: no cover - CLI wiring
    handle_tag_delete(client, output_format, tag_id)


@tag.command("attach")
@click.argument("tag_id")
@click.argument("note_id")
@format_option
@with_client
def tag_attach(
    client: JoplinClient, output_format: str, tag_id: str, note_id: str
) -> None:  # pragma: no cover - CLI wiring
    handle_tag_attach(client, output_format, tag_id, note_id)


@tag.command("detach")
@click.argument("tag_id")
@click.argument("note_id")
@format_option
@with_client
def tag_detach(
    client: JoplinClient, output_format: str, tag_id: str, note_id: str
) -> None:  # pragma: no cover - CLI wiring
    handle_tag_detach(client, output_format, tag_id, note_id)


@cli.group()
def resource() -> None:
    """Resource commands."""


@resource.command("list")
@click.option("--limit", type=int, help="Limit number of resources.")
@format_option
@with_client
def resource_list(
    client: JoplinClient, output_format: str, limit: Optional[int]
) -> None:  # pragma: no cover - CLI wiring
    handle_resource_list(client, output_format, limit)


@resource.command("show")
@click.argument("resource_id")
@format_option
@with_client
def resource_show(
    client: JoplinClient, output_format: str, resource_id: str
) -> None:  # pragma: no cover - CLI wiring
    handle_resource_show(client, output_format, resource_id)


@resource.command("create")
@click.option("--file", "file_path_str", required=True, help="Path to the file to upload.")
@click.option("--title", help="Resource title.")
@click.option("--mime", help="MIME type override.")
@click.option("--data", help="Additional JSON object merged into the resource properties.")
@format_option
@with_client
def resource_create(
    client: JoplinClient,
    output_format: str,
    file_path_str: str,
    title: Optional[str],
    mime: Optional[str],
    data: Optional[str],
) -> None:  # pragma: no cover - CLI wiring
    handle_resource_create(client, output_format, file_path_str, title, mime, data)


@resource.command("update")
@click.argument("resource_id")
@click.option("--file", "file_path_str", help="Path to the file to upload.")
@click.option("--title", help="Resource title.")
@click.option("--mime", help="MIME type override.")
@click.option("--data", help="Additional JSON object merged into the resource properties.")
@format_option
@with_client
def resource_update(
    client: JoplinClient,
    output_format: str,
    resource_id: str,
    file_path_str: Optional[str],
    title: Optional[str],
    mime: Optional[str],
    data: Optional[str],
) -> None:  # pragma: no cover - CLI wiring
    handle_resource_update(client, output_format, resource_id, file_path_str, title, mime, data)


@resource.command("delete")
@click.argument("resource_id")
@format_option
@with_client
def resource_delete(
    client: JoplinClient, output_format: str, resource_id: str
) -> None:  # pragma: no cover - CLI wiring
    handle_resource_delete(client, output_format, resource_id)


def main() -> None:  # pragma: no cover - console entry point
    """Console entry hook for setuptools."""
    cli()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
