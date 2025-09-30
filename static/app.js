document.addEventListener("DOMContentLoaded", () => {
  // Tab navigation
  const tabs = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.tab).classList.add("active");
      if (tab.dataset.tab === "list") {
        fetchProducts();
      }
    });
  });

  // Set default tab
  document.querySelector('.tab-button[data-tab="create"]').click();

  // Create Product Form
  document
    .getElementById("create-product-form")
    .addEventListener("submit", async (e) => {
      e.preventDefault();
      const form = e.target;
      const formData = new FormData(form);
      try {
        const response = await fetch("/products", {
          method: "POST",
          body: formData,
        });
        const result = await response.json();
        const resultDiv = document.getElementById("create-result");
        if (response.ok) {
          resultDiv.innerHTML = `<p class="text-green-600">Product created: ${result.name}</p>`;
          form.reset();
        } else {
          resultDiv.innerHTML = `<p class="error">Error: ${result.error}</p>`;
        }
      } catch (error) {
        document.getElementById(
          "create-result"
        ).innerHTML = `<p class="error">Error: ${error.message}</p>`;
      }
    });

  // List Products
  async function fetchProducts(category = "") {
    try {
      const url = category
        ? `/products?category=${encodeURIComponent(category)}`
        : "/products";
      const response = await fetch(url);
      const products = await response.json();
      const listDiv = document.getElementById("products-list");
      if (response.ok && products.length) {
        let html =
          "<table><tr><th>ID</th><th>Name</th><th>Category</th><th>Price</th><th>Images</th><th>Action</th></tr>";
        products.forEach((p) => {
          html += `<tr>
                        <td>${p.product_id}</td>
                        <td>${p.name}</td>
                        <td>${p.category || "-"}</td>
                        <td>${p.price ? `$${p.price.toFixed(2)}` : "-"}</td>
                        <td>${p.reference_images_count}</td>
                        <td><button onclick="deleteProduct('${
                          p.product_id
                        }')" class="text-red-500 hover:underline">Delete</button></td>
                    </tr>`;
        });
        html += "</table>";
        listDiv.innerHTML = html;
      } else {
        listDiv.innerHTML = "<p>No products found.</p>";
      }
    } catch (error) {
      document.getElementById(
        "products-list"
      ).innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
  }

  // Category Filter
  document.getElementById("category-filter").addEventListener("input", (e) => {
    fetchProducts(e.target.value);
  });

  // Delete Product
  window.deleteProduct = async (productId) => {
    if (!confirm(`Delete product ${productId}?`)) return;
    try {
      const response = await fetch(`/products/${productId}`, {
        method: "DELETE",
      });
      const result = await response.json();
      if (response.ok) {
        fetchProducts(document.getElementById("category-filter").value);
        alert(result.message);
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  // Count Products Form
  document
    .getElementById("count-form")
    .addEventListener("submit", async (e) => {
      e.preventDefault();
      const form = e.target;
      const formData = new FormData(form);
      try {
        const response = await fetch("/count", {
          method: "POST",
          body: formData,
        });
        const result = await response.json();
        const resultDiv = document.getElementById("count-result");
        if (response.ok) {
          let html = `<h3 class="text-xl font-semibold">Results for ${result.image_name}</h3>`;
          html += `<p>Processing Time: ${result.processing_time.toFixed(
            2
          )}s</p>`;
          html += `<p>Total Products: ${result.summary.total_products_detected}</p>`;
          html += `<p>Unique Products: ${result.summary.unique_products_detected}</p>`;
          html += `<p>Detection Rate: ${(
            result.summary.detection_rate * 100
          ).toFixed(2)}%</p>`;
          html +=
            "<table><tr><th>Product</th><th>Count</th><th>Avg Confidence</th></tr>";
          result.product_counts.forEach((pc) => {
            html += `<tr><td>${pc.product_name}</td><td>${pc.count}</td><td>${(
              pc.avg_confidence * 100
            ).toFixed(2)}%</td></tr>`;
          });
          html += "</table>";
          if (result.visualization_url) {
            html += `<img src="${result.visualization_url}" alt="Visualization" class="mt-4 max-w-full">`;
          }
          resultDiv.innerHTML = html;
          form.reset();
        } else {
          resultDiv.innerHTML = `<p class="error">Error: ${result.error}</p>`;
        }
      } catch (error) {
        document.getElementById(
          "count-result"
        ).innerHTML = `<p class="error">Error: ${error.message}</p>`;
      }
    });

  // Batch Count Form
  document
    .getElementById("batch-count-form")
    .addEventListener("submit", async (e) => {
      e.preventDefault();
      const form = e.target;
      const formData = new FormData(form);
      try {
        const response = await fetch("/count/batch", {
          method: "POST",
          body: formData,
        });
        const result = await response.json();
        const resultDiv = document.getElementById("batch-count-result");
        if (response.ok) {
          let html = `<h3 class="text-xl font-semibold">Batch Results</h3>`;
          html += `<p>Total Images: ${result.total_images}</p>`;
          html += `<p>Successful: ${result.successful_counts}</p>`;
          html += `<p>Failed: ${result.failed_counts}</p>`;
          html += `<p>Total Processing Time: ${result.total_processing_time.toFixed(
            2
          )}s</p>`;
          html += `<p>Total Products: ${result.aggregated_summary.total_products_detected}</p>`;
          html += `<p>Unique Products: ${result.aggregated_summary.unique_products_detected}</p>`;
          html += `<table><tr><th>Product</th><th>Total Count</th><th>Avg Confidence</th></tr>`;
          result.aggregated_product_counts.forEach((pc) => {
            html += `<tr><td>${pc.product_name}</td><td>${pc.count}</td><td>${(
              pc.avg_confidence * 100
            ).toFixed(2)}%</td></tr>`;
          });
          html += "</table>";
          resultDiv.innerHTML = html;
          form.reset();
        } else {
          resultDiv.innerHTML = `<p class="error">Error: ${result.error}</p>`;
        }
      } catch (error) {
        document.getElementById(
          "batch-count-result"
        ).innerHTML = `<p class="error">Error: ${error.message}</p>`;
      }
    });

  // Get Stats
  document.getElementById("get-stats").addEventListener("click", async () => {
    try {
      const response = await fetch("/stats");
      const result = await response.json();
      const resultDiv = document.getElementById("stats-result");
      if (response.ok) {
        let html = `<h3 class="text-xl font-semibold">System Statistics</h3>`;
        html += `<p>Last Updated: ${result.last_updated}</p>`;
        html += `<p>Catalog Size: ${result.catalog_stats.total_products}</p>`;
        html += `<p>System Info: ${JSON.stringify(result.system_info)}</p>`;
        resultDiv.innerHTML = html;
      } else {
        resultDiv.innerHTML = `<p class="error">Error: ${result.error}</p>`;
      }
    } catch (error) {
      document.getElementById(
        "stats-result"
      ).innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
  });

  // Export Catalog
  document
    .getElementById("export-catalog")
    .addEventListener("click", async () => {
      const format = document.getElementById("export-format").value;
      try {
        const response = await fetch(`/catalog/export?format=${format}`);
        if (response.ok) {
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `catalog_export_${Date.now()}.${format}`;
          a.click();
          window.URL.revokeObjectURL(url);
        } else {
          const result = await response.json();
          document.getElementById(
            "stats-result"
          ).innerHTML = `<p class="error">Error: ${result.error}</p>`;
        }
      } catch (error) {
        document.getElementById(
          "stats-result"
        ).innerHTML = `<p class="error">Error: ${error.message}</p>`;
      }
    });
});
