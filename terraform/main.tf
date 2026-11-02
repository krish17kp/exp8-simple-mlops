terraform {
  required_providers {
    docker = { source = "kreuzwerker/docker", version = "~> 3.0" }
  }
}
provider "docker" {}
resource "docker_image" "delivery_ml" {
  name = "delivery-mlops:latest"
  build { context = ".." }
}
resource "docker_container" "delivery_ml" {
  name  = "delivery-mlops"
  image = docker_image.delivery_ml.image_id
  ports {
    internal = 8501
    external = 8501
  }
}
