# Requirements Document

## Introduction

This feature enables automatic task detection and management from text input. The system analyzes text (from any source, including the existing speech-to-text system) using a local LLM to determine if the text represents a task. Detected tasks are stored in a persistent task list and exposed via Model Context Protocol (MCP) for integration with other AI tools and workflows.

## Glossary

- **Task Detection System**: The component that analyzes text input and determines whether it represents an actionable task
- **Task List Manager**: The component responsible for storing, retrieving, and managing tasks
- **LLM Classifier**: The local language model (via Ollama) that performs task classification
- **MCP Server**: The Model Context Protocol server that exposes task list operations to external tools
- **Task Item**: A structured data object containing task description, metadata, and status
- **Text Input Source**: Any component that provides text to the Task Detection System (e.g., speech-to-text, CLI, API)

## Requirements

### Requirement 1

**User Story:** As a user, I want to provide text input from any source and have the system automatically detect if it contains a task, so that I can capture tasks without manual categorization.

#### Acceptance Criteria

1. WHEN text input is received by the Task Detection System, THE Task Detection System SHALL pass the text to the LLM Classifier for analysis
2. WHEN the LLM Classifier analyzes text, THE LLM Classifier SHALL return a classification indicating whether the text represents a task with a confidence score
3. IF the LLM Classifier determines text represents a task with confidence above the configured threshold, THEN THE Task Detection System SHALL extract task details including description, priority, and due date if present
4. WHEN a task is detected, THE Task Detection System SHALL pass the extracted task details to the Task List Manager for storage
5. WHEN task detection completes, THE Task Detection System SHALL return a result indicating whether a task was detected and stored

### Requirement 2

**User Story:** As a user, I want the system to use a local LLM via Ollama for task detection, so that my data remains private and the system works offline.

#### Acceptance Criteria

1. WHEN the LLM Classifier initializes, THE LLM Classifier SHALL connect to the local Ollama service using the configured connection parameters
2. WHEN the Ollama service is unavailable, THE LLM Classifier SHALL raise an initialization error with a descriptive message
3. WHEN the LLM Classifier sends a classification request, THE LLM Classifier SHALL use a prompt template that instructs the model to identify tasks and extract relevant details
4. WHEN the LLM responds, THE LLM Classifier SHALL parse the response into a structured classification result with task details
5. WHERE the user configures a specific Ollama model, THE LLM Classifier SHALL use the specified model for all classification requests

### Requirement 3

**User Story:** As a user, I want tasks to be stored persistently in a local file, so that my task list survives application restarts and I can access it from other tools.

#### Acceptance Criteria

1. WHEN the Task List Manager initializes, THE Task List Manager SHALL load existing tasks from the configured storage file path
2. IF the storage file does not exist, THEN THE Task List Manager SHALL create a new empty task list
3. WHEN a new task is added, THE Task List Manager SHALL append the task to the in-memory task list and persist it to the storage file
4. WHEN tasks are persisted, THE Task List Manager SHALL use a structured format that preserves task metadata including creation time, status, priority, and description
5. WHEN the storage file is corrupted or invalid, THE Task List Manager SHALL log an error and initialize with an empty task list

### Requirement 4

**User Story:** As a user, I want to access and manage my task list through MCP, so that I can integrate task management with other AI tools and workflows.

#### Acceptance Criteria

1. WHEN the MCP Server starts, THE MCP Server SHALL expose tools for listing tasks, adding tasks, updating task status, and deleting tasks
2. WHEN an MCP client requests the task list, THE MCP Server SHALL retrieve all tasks from the Task List Manager and return them in a structured format
3. WHEN an MCP client adds a task via the MCP Server, THE MCP Server SHALL validate the task data and pass it to the Task List Manager for storage
4. WHEN an MCP client updates a task status, THE MCP Server SHALL locate the task by identifier and update its status through the Task List Manager
5. WHEN an MCP client deletes a task, THE MCP Server SHALL remove the task from the Task List Manager and confirm the deletion

### Requirement 5

**User Story:** As a user, I want the system to handle edge cases gracefully, so that temporary issues don't cause data loss or system crashes.

#### Acceptance Criteria

1. WHEN the LLM Classifier encounters a timeout from Ollama, THE LLM Classifier SHALL retry the request up to the configured maximum retry count
2. IF all retry attempts fail, THEN THE Task Detection System SHALL return a classification failure result without crashing
3. WHEN the Task List Manager encounters a file write error, THE Task List Manager SHALL log the error and maintain the in-memory task list
4. WHEN invalid text input is received, THE Task Detection System SHALL handle the input gracefully and return a no-task-detected result
5. WHEN the MCP Server receives malformed requests, THE MCP Server SHALL return appropriate error responses without terminating the server

### Requirement 6

**User Story:** As a user, I want to configure the task detection behavior, so that I can tune the system to my preferences and hardware constraints.

#### Acceptance Criteria

1. THE Task Detection System SHALL support configuration of the confidence threshold for task classification
2. THE LLM Classifier SHALL support configuration of the Ollama model name, connection URL, and timeout duration
3. THE Task List Manager SHALL support configuration of the storage file path and format
4. THE MCP Server SHALL support configuration of the server host, port, and authentication settings
5. WHERE configuration values are not provided, THE Task Detection System SHALL use sensible defaults optimized for 8GB GPU systems

### Requirement 7

**User Story:** As a developer, I want the task detection system to integrate seamlessly with the existing speech-to-text pipeline, so that voice input can automatically create tasks.

#### Acceptance Criteria

1. THE Task Detection System SHALL accept text input through a standardized interface that the speech-to-text pipeline can invoke
2. WHEN the speech-to-text pipeline produces a transcription, THE speech-to-text pipeline SHALL have the option to pass the text to the Task Detection System
3. WHEN task detection is triggered from speech input, THE Task Detection System SHALL process the text asynchronously to avoid blocking the speech pipeline
4. WHEN a task is detected from speech input, THE Task Detection System SHALL provide feedback through the configured notification mechanism
5. THE Task Detection System SHALL maintain independence from the speech-to-text module to allow text input from other sources

### Requirement 8

**User Story:** As a user, I want visibility into task detection performance and accuracy, so that I can understand system behavior and optimize configuration.

#### Acceptance Criteria

1. WHEN task detection completes, THE Task Detection System SHALL log the classification result including confidence score and processing time
2. WHEN the LLM Classifier processes a request, THE LLM Classifier SHALL track and report the inference time
3. THE Task List Manager SHALL maintain statistics on total tasks, completed tasks, and pending tasks
4. THE Task Detection System SHALL expose performance metrics through the existing performance monitoring system
5. WHERE verbose logging is enabled, THE Task Detection System SHALL log the full LLM prompt and response for debugging purposes
