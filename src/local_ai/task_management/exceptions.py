"""Custom exceptions for task management functionality."""


class TaskManagementError(Exception):
    """Base exception for task management errors."""

    pass


class DatabaseError(TaskManagementError):
    """Exception raised for database related errors."""

    pass


class SchemaError(DatabaseError):
    """Exception raised for database schema errors."""

    pass


class TaskNotFoundError(TaskManagementError):
    """Exception raised when a task is not found."""

    pass


class ClassificationError(TaskManagementError):
    """Exception raised for LLM classification errors."""

    pass
