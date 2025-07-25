// agent-invoke API Route

export async function POST(request: Request) {
    console.log("Received request...");

    try {
        // Parse the incoming request body
        const requestBody = await request.json();
        console.log("Request Body:", requestBody);

        // Forward the text to the Python server
        const response = await fetch('http://127.0.0.1:5000/agent-invoke', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: requestBody.text }),
        });

        if (response.ok) {
            // Parse and return the response from the Python server
            const result = await response.json();
            return new Response(
                JSON.stringify({ message: "Communication successful!", data: result }),
                { status: 200, headers: { 'Content-Type': 'application/json' } }
            );
        } else {
            const error = await response.json();
            console.error("Error from Python server:", error);
            return new Response(
                JSON.stringify({ error: "Error from Python server", details: error }),
                { status: response.status, headers: { 'Content-Type': 'application/json' } }
            );
        }
    } catch (error) {
        console.error("Internal Server Error:", error);
        return new Response(
            JSON.stringify({ message: "Internal server error", details: String(error) }),
            { status: 500, headers: { 'Content-Type': 'application/json' } }
        );
    }
}
