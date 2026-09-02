import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

/** Minimal Java Micronaut contract for the Cline/JShell adapter. */
class JavaMicronautAgent {
    private final Map<String, Object> state = new LinkedHashMap<>();

    public JavaMicronautAgent() {
        state.put("phase", "Pop");
        state.put("created", Instant.now().toString());
    }

    public Map<String, Object> pop() {
        state.put("phase", "Pop");
        return snapshot();
    }

    public Map<String, Object> sek(String operation) {
        state.put("phase", "Sek");
        state.put("operation", operation);
        return snapshot();
    }

    public Map<String, Object> chen(boolean valid) {
        state.put("phase", "Chen");
        state.put("valid", valid);
        return snapshot();
    }

    public Map<String, Object> xul() {
        state.put("phase", "Xul");
        return snapshot();
    }

    public Map<String, Object> snapshot() {
        return new LinkedHashMap<>(state);
    }
}
