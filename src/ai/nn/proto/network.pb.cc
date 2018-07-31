// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: network.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "network.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace ncooper {
namespace ai {
namespace nn {
class NeuronDefaultTypeInternal {
public:
 ::google::protobuf::internal::ExplicitlyConstructed<Neuron>
     _instance;
} _Neuron_default_instance_;

namespace protobuf_network_2eproto {


namespace {

::google::protobuf::Metadata file_level_metadata[1];

}  // namespace

PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::ParseTableField
    const TableStruct::entries[] GOOGLE_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  {0, 0, 0, ::google::protobuf::internal::kInvalidMask, 0, 0},
};

PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::AuxillaryParseTableField
    const TableStruct::aux[] GOOGLE_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ::google::protobuf::internal::AuxillaryParseTableField(),
};
PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::ParseTable const
    TableStruct::schema[] GOOGLE_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { NULL, NULL, 0, -1, -1, -1, -1, NULL, false },
};

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Neuron, _has_bits_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Neuron, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Neuron, id_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Neuron, weights_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Neuron, bias_),
  0,
  ~0u,
  1,
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 8, sizeof(Neuron)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&_Neuron_default_instance_),
};

namespace {

void protobuf_AssignDescriptors() {
  AddDescriptors();
  ::google::protobuf::MessageFactory* factory = NULL;
  AssignDescriptors(
      "network.proto", schemas, file_default_instances, TableStruct::offsets, factory,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

}  // namespace
void TableStruct::InitDefaultsImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::internal::InitProtobufDefaults();
  _Neuron_default_instance_._instance.DefaultConstruct();
  ::google::protobuf::internal::OnShutdownDestroyMessage(
      &_Neuron_default_instance_);}

void InitDefaults() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &TableStruct::InitDefaultsImpl);
}
namespace {
void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\rnetwork.proto\022\rncooper.ai.nn\"3\n\006Neuron"
      "\022\n\n\002id\030\001 \002(\005\022\017\n\007weights\030\002 \003(\002\022\014\n\004bias\030\003 "
      "\001(\002"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 83);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "network.proto", &protobuf_RegisterTypes);
}
} // anonymous namespace

void AddDescriptors() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;

}  // namespace protobuf_network_2eproto


// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Neuron::kIdFieldNumber;
const int Neuron::kWeightsFieldNumber;
const int Neuron::kBiasFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Neuron::Neuron()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    protobuf_network_2eproto::InitDefaults();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:ncooper.ai.nn.Neuron)
}
Neuron::Neuron(const Neuron& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      _has_bits_(from._has_bits_),
      _cached_size_(0),
      weights_(from.weights_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::memcpy(&id_, &from.id_,
    static_cast<size_t>(reinterpret_cast<char*>(&bias_) -
    reinterpret_cast<char*>(&id_)) + sizeof(bias_));
  // @@protoc_insertion_point(copy_constructor:ncooper.ai.nn.Neuron)
}

void Neuron::SharedCtor() {
  _cached_size_ = 0;
  ::memset(&id_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&bias_) -
      reinterpret_cast<char*>(&id_)) + sizeof(bias_));
}

Neuron::~Neuron() {
  // @@protoc_insertion_point(destructor:ncooper.ai.nn.Neuron)
  SharedDtor();
}

void Neuron::SharedDtor() {
}

void Neuron::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Neuron::descriptor() {
  protobuf_network_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_network_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const Neuron& Neuron::default_instance() {
  protobuf_network_2eproto::InitDefaults();
  return *internal_default_instance();
}

Neuron* Neuron::New(::google::protobuf::Arena* arena) const {
  Neuron* n = new Neuron;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void Neuron::Clear() {
// @@protoc_insertion_point(message_clear_start:ncooper.ai.nn.Neuron)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  weights_.Clear();
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 3u) {
    ::memset(&id_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&bias_) -
        reinterpret_cast<char*>(&id_)) + sizeof(bias_));
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear();
}

bool Neuron::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:ncooper.ai.nn.Neuron)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required int32 id = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(8u /* 8 & 0xFF */)) {
          set_has_id();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &id_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated float weights = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(21u /* 21 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 21u, input, this->mutable_weights())));
        } else if (
            static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_weights())));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // optional float bias = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(29u /* 29 & 0xFF */)) {
          set_has_bias();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &bias_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:ncooper.ai.nn.Neuron)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:ncooper.ai.nn.Neuron)
  return false;
#undef DO_
}

void Neuron::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:ncooper.ai.nn.Neuron)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required int32 id = 1;
  if (cached_has_bits & 0x00000001u) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(1, this->id(), output);
  }

  // repeated float weights = 2;
  for (int i = 0, n = this->weights_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(
      2, this->weights(i), output);
  }

  // optional float bias = 3;
  if (cached_has_bits & 0x00000002u) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(3, this->bias(), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:ncooper.ai.nn.Neuron)
}

::google::protobuf::uint8* Neuron::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:ncooper.ai.nn.Neuron)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required int32 id = 1;
  if (cached_has_bits & 0x00000001u) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(1, this->id(), target);
  }

  // repeated float weights = 2;
  target = ::google::protobuf::internal::WireFormatLite::
    WriteFloatToArray(2, this->weights_, target);

  // optional float bias = 3;
  if (cached_has_bits & 0x00000002u) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(3, this->bias(), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:ncooper.ai.nn.Neuron)
  return target;
}

size_t Neuron::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:ncooper.ai.nn.Neuron)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  // required int32 id = 1;
  if (has_id()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->id());
  }
  // repeated float weights = 2;
  {
    unsigned int count = static_cast<unsigned int>(this->weights_size());
    size_t data_size = 4UL * count;
    total_size += 1 *
                  ::google::protobuf::internal::FromIntSize(this->weights_size());
    total_size += data_size;
  }

  // optional float bias = 3;
  if (has_bias()) {
    total_size += 1 + 4;
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Neuron::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:ncooper.ai.nn.Neuron)
  GOOGLE_DCHECK_NE(&from, this);
  const Neuron* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const Neuron>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:ncooper.ai.nn.Neuron)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:ncooper.ai.nn.Neuron)
    MergeFrom(*source);
  }
}

void Neuron::MergeFrom(const Neuron& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:ncooper.ai.nn.Neuron)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  weights_.MergeFrom(from.weights_);
  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 3u) {
    if (cached_has_bits & 0x00000001u) {
      id_ = from.id_;
    }
    if (cached_has_bits & 0x00000002u) {
      bias_ = from.bias_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void Neuron::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:ncooper.ai.nn.Neuron)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Neuron::CopyFrom(const Neuron& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:ncooper.ai.nn.Neuron)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Neuron::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000001) != 0x00000001) return false;
  return true;
}

void Neuron::Swap(Neuron* other) {
  if (other == this) return;
  InternalSwap(other);
}
void Neuron::InternalSwap(Neuron* other) {
  using std::swap;
  weights_.InternalSwap(&other->weights_);
  swap(id_, other->id_);
  swap(bias_, other->bias_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata Neuron::GetMetadata() const {
  protobuf_network_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_network_2eproto::file_level_metadata[kIndexInFileMessages];
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// Neuron

// required int32 id = 1;
bool Neuron::has_id() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
void Neuron::set_has_id() {
  _has_bits_[0] |= 0x00000001u;
}
void Neuron::clear_has_id() {
  _has_bits_[0] &= ~0x00000001u;
}
void Neuron::clear_id() {
  id_ = 0;
  clear_has_id();
}
::google::protobuf::int32 Neuron::id() const {
  // @@protoc_insertion_point(field_get:ncooper.ai.nn.Neuron.id)
  return id_;
}
void Neuron::set_id(::google::protobuf::int32 value) {
  set_has_id();
  id_ = value;
  // @@protoc_insertion_point(field_set:ncooper.ai.nn.Neuron.id)
}

// repeated float weights = 2;
int Neuron::weights_size() const {
  return weights_.size();
}
void Neuron::clear_weights() {
  weights_.Clear();
}
float Neuron::weights(int index) const {
  // @@protoc_insertion_point(field_get:ncooper.ai.nn.Neuron.weights)
  return weights_.Get(index);
}
void Neuron::set_weights(int index, float value) {
  weights_.Set(index, value);
  // @@protoc_insertion_point(field_set:ncooper.ai.nn.Neuron.weights)
}
void Neuron::add_weights(float value) {
  weights_.Add(value);
  // @@protoc_insertion_point(field_add:ncooper.ai.nn.Neuron.weights)
}
const ::google::protobuf::RepeatedField< float >&
Neuron::weights() const {
  // @@protoc_insertion_point(field_list:ncooper.ai.nn.Neuron.weights)
  return weights_;
}
::google::protobuf::RepeatedField< float >*
Neuron::mutable_weights() {
  // @@protoc_insertion_point(field_mutable_list:ncooper.ai.nn.Neuron.weights)
  return &weights_;
}

// optional float bias = 3;
bool Neuron::has_bias() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
void Neuron::set_has_bias() {
  _has_bits_[0] |= 0x00000002u;
}
void Neuron::clear_has_bias() {
  _has_bits_[0] &= ~0x00000002u;
}
void Neuron::clear_bias() {
  bias_ = 0;
  clear_has_bias();
}
float Neuron::bias() const {
  // @@protoc_insertion_point(field_get:ncooper.ai.nn.Neuron.bias)
  return bias_;
}
void Neuron::set_bias(float value) {
  set_has_bias();
  bias_ = value;
  // @@protoc_insertion_point(field_set:ncooper.ai.nn.Neuron.bias)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace nn
}  // namespace ai
}  // namespace ncooper

// @@protoc_insertion_point(global_scope)
