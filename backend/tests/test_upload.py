import unittest
from contextlib import redirect_stdout
from io import BytesIO, StringIO

from httpx import ASGITransport, AsyncClient

from backend.app import app


class UploadFilesTestCase(unittest.IsolatedAsyncioTestCase):
    """Regression tests for the multipart upload endpoint."""

    async def test_upload_prints_and_returns_all_filenames(self) -> None:
        """Verify multiple filenames are printed and returned in order.

        Returns:
            None. Assertions fail the test when endpoint behavior changes.
        """
        output = StringIO()
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            with redirect_stdout(output):
                response = await client.post(
                    "/upload/",
                    files=[
                        ("files", ("notes.txt", BytesIO(b"notes"), "text/plain")),
                        (
                            "files",
                            ("report.pdf", BytesIO(b"pdf"), "application/pdf"),
                        ),
                    ],
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"filenames": ["notes.txt", "report.pdf"]},
        )
        self.assertEqual(output.getvalue().splitlines(), ["notes.txt", "report.pdf"])


if __name__ == "__main__":
    unittest.main()
