variable "instance_number" {
  type        = number
  description = "The number of instances to create"
  default     = 1

  validation {
    condition     = var.instance_number >= 1 && var.instance_number <= 10
    error_message = "The instance number has to be between 1 and 10 (inclusive)."
  }
}
