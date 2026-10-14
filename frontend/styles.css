const imageInput =
    document.getElementById("imageInput");

const predictButton =
    document.getElementById("predictButton");

const status =
    document.getElementById("status");

const results =
    document.getElementById("results");


// ---------------------------------------------------------
// IMPORTANT
//
// Because CloudFront routes /api/* to API Gateway,
// the browser does not need to know the API Gateway URL.
// ---------------------------------------------------------

const API_URL = "/api/predict";


let selectedFile = null;


// ---------------------------------------------------------
// Image selection
// ---------------------------------------------------------

imageInput.addEventListener(
    "change",
    () => {

        selectedFile =
            imageInput.files[0];

        predictButton.disabled =
            !selectedFile;

        results.classList.add(
            "hidden"
        );

        if (selectedFile) {

            status.textContent =
                `Selected: ${selectedFile.name}`;
        }
    }
);


// ---------------------------------------------------------
// Resize/compress phone image
// ---------------------------------------------------------

async function prepareImage(file) {

    const bitmap =
        await createImageBitmap(file);

    const MAX_SIZE = 1600;

    let width =
        bitmap.width;

    let height =
        bitmap.height;


    if (width > MAX_SIZE ||
        height > MAX_SIZE) {

        const scale =
            Math.min(
                MAX_SIZE / width,
                MAX_SIZE / height
            );

        width =
            Math.round(width * scale);

        height =
            Math.round(height * scale);
    }


    const canvas =
        document.createElement(
            "canvas"
        );

    canvas.width =
        width;

    canvas.height =
        height;


    const ctx =
        canvas.getContext(
            "2d"
        );

    ctx.drawImage(
        bitmap,
        0,
        0,
        width,
        height
    );


    return new Promise(
        resolve => {

            canvas.toBlob(
                blob => resolve(blob),
                "image/jpeg",
                0.85
            );

        }
    );
}


// ---------------------------------------------------------
// Convert Blob → base64
// ---------------------------------------------------------

function blobToBase64(blob) {

    return new Promise(
        (resolve, reject) => {

            const reader =
                new FileReader();

            reader.onload = () => {

                const result =
                    reader.result;

                // Remove:
                // data:image/jpeg;base64,

                const base64 =
                    result.split(",")[1];

                resolve(base64);
            };

            reader.onerror =
                reject;

            reader.readAsDataURL(
                blob
            );
        }
    );
}


// ---------------------------------------------------------
// Display top 3
// ---------------------------------------------------------

function renderTop3(
    container,
    predictions
) {

    container.innerHTML = "";

    predictions.forEach(
        item => {

            const row =
                document.createElement(
                    "div"
                );

            const probability =
                (
                    item.probability * 100
                ).toFixed(1);


            row.innerHTML = `
                <span>
                    ${item.class}
                </span>

                <span>
                    ${probability}%
                </span>
            `;

            container.appendChild(
                row
            );
        }
    );
}


// ---------------------------------------------------------
// Display one model
// ---------------------------------------------------------

function renderModel(
    prefix,
    data,
    imageBase64
) {

    document.getElementById(
        `${prefix}Image`
    ).src =
        `data:image/png;base64,${imageBase64}`;


    document.getElementById(
        `${prefix}Prediction`
    ).textContent =
        `Prediction: ${data.prediction}`;


    renderTop3(
        document.getElementById(
            `${prefix}Top3`
        ),
        data.top3
    );
}


// ---------------------------------------------------------
// Prediction
// ---------------------------------------------------------

predictButton.addEventListener(
    "click",
    async () => {

        if (!selectedFile) {
            return;
        }


        predictButton.disabled =
            true;

        status.textContent =
            "Processing image and running models...";


        try {

            const prepared =
                await prepareImage(
                    selectedFile
                );

            const base64 =
                await blobToBase64(
                    prepared
                );


            const response =
                await fetch(
                    API_URL,
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            image: base64
                        })
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Prediction failed"
                );
            }


            renderModel(
                "raw",
                data.raw,
                data.images.raw
            );


            renderModel(
                "filtered",
                data.filtered,
                data.images.filtered
            );


            renderModel(
                "edges",
                data.edges,
                data.images.edges
            );


            results.classList.remove(
                "hidden"
            );


            status.textContent =
                "Prediction complete.";

        }
        catch (error) {

            console.error(
                error
            );

            status.textContent =
                `Error: ${error.message}`;

        }
        finally {

            predictButton.disabled =
                false;
        }

    }
);